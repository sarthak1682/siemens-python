import os
import sys
import argparse
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, AllChem, DataStructs

# Set non-interactive backend for saving plots without a display
plt.switch_backend('agg')



def set_seed(seed: int):
    print(f"--- Setting Global Seed to {seed} ---")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def download_data():
    if not os.path.exists('gdb9.sdf'):
        print("Downloading QM9 dataset...")
        # Simulating !wget and !tar
        subprocess.run(["wget", "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"], check=True)
        subprocess.run(["tar", "-xvf", "gdb9.tar.gz"], check=True)
        print("Download complete.")
    else:
        print("gdb9.sdf already exists. Skipping download.")

def grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return float(np.sqrt(total)) if total > 0 else 0.0

def calculate_tanimoto_stats(gen_smiles, train_smiles, n_sample=1000):
    print("Calculating Tanimoto/Fingerprint stats...")
    
    # Filter valid molecules
    gen_mols = [Chem.MolFromSmiles(s) for s in gen_smiles if Chem.MolFromSmiles(s)]
    train_mols = [Chem.MolFromSmiles(s) for s in train_smiles if Chem.MolFromSmiles(s)]
    
    # Shuffle and slice to save time
    random.shuffle(gen_mols)
    random.shuffle(train_mols)
    gen_mols = gen_mols[:n_sample]
    train_mols = train_mols[:n_sample]
    
    if len(gen_mols) < 2:
        return 0.0, 0.0, 0.0

    # Get Fingerprints
    gen_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in gen_mols]
    train_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in train_mols]
    
    # Internal Similarity (Mean Pairwise Tanimoto)
    sims = []
    for i in range(len(gen_fps)):
        for j in range(i+1, len(gen_fps)):
            sims.append(DataStructs.TanimotoSimilarity(gen_fps[i], gen_fps[j]))
            if len(sims) > 5000: break
        if len(sims) > 5000: break
            
    avg_internal_sim = np.mean(sims) if sims else 0.0
    
    # Novelty Score (Max Sim vs Train) & % Novel
    max_sims = []
    for gfp in gen_fps:
        m_sim = max([DataStructs.TanimotoSimilarity(gfp, tfp) for tfp in train_fps])
        max_sims.append(m_sim)
    
    avg_novelty_score = np.mean(max_sims) if max_sims else 0.0
    percent_novel = sum(1 for s in max_sims if s < 0.4) / len(max_sims) * 100 if max_sims else 0.0
    
    return avg_internal_sim, avg_novelty_score, percent_novel

class SMILESTokenizer:
    def __init__(self, smiles_list):
        chars = set()
        for s in smiles_list:
            chars.update(s)
        self.char_to_idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
        for i, c in enumerate(sorted(chars)):
            self.char_to_idx[c] = i + 3
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
    
    def encode(self, smiles, max_len):
        enc = [self.char_to_idx['<SOS>']] + [self.char_to_idx.get(c, 0) for c in smiles] + [self.char_to_idx['<EOS>']]
        return enc[:max_len] + [0] * max(0, max_len - len(enc))
    
    def decode(self, indices):
        chars = []
        for i in indices:
            c = self.idx_to_char.get(i, '')
            if c == '<EOS>':
                break
            if c not in ['<PAD>', '<SOS>']:
                chars.append(c)
        return ''.join(chars)

class SMILESDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_len):
        self.data = [tokenizer.encode(s, max_len) for s in smiles_list]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = torch.LongTensor(self.data[idx])
        return seq, seq[:-1], seq[1:]




class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
    
    def sample_h(self, v):
        h_prob = torch.sigmoid(F.linear(v, self.W.t(), self.h_bias))
        h_sample = torch.bernoulli(h_prob)
        return h_prob, h_sample
    
    def sample_v(self, h):
        v_prob = torch.sigmoid(F.linear(h, self.W, self.v_bias))
        v_sample = torch.bernoulli(v_prob)
        return v_prob, v_sample
    
    def forward(self, v):
        h_prob, _ = self.sample_h(v)
        return h_prob
    
    def free_energy(self, v):
        vbias_term = (v * self.v_bias).sum(1)
        wx_b = F.linear(v, self.W.t(), self.h_bias)
        hidden_term = F.softplus(wx_b).sum(1)
        return -vbias_term - hidden_term

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, latent_dim, max_len, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.pos_enc = nn.Parameter(torch.randn(1, max_len, emb_dim) * 0.01)
        self.conv = nn.Conv1d(emb_dim, hidden_dim, 9, padding=4)
        self.highway_gate = nn.Linear(hidden_dim, hidden_dim)
        self.highway_proj = nn.Linear(emb_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, 8, hidden_dim*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=5)
        self.num_subsequences = 3
        self.fc_logits = nn.Linear(hidden_dim * (2 + self.num_subsequences), latent_dim)

    def forward(self, x):
        mask = x.eq(self.pad_idx)
        emb = self.embedding(x) + self.pos_enc[:, :x.size(1), :]
        conv_out = F.relu(self.conv(emb.transpose(1, 2))).transpose(1, 2)
        gate = torch.sigmoid(self.highway_gate(conv_out))
        h = gate * conv_out + (1 - gate) * self.highway_proj(emb)
        enc = self.transformer(h, src_key_padding_mask=mask)

        masked = enc.masked_fill(mask.unsqueeze(-1), 0.0)
        real_len = (~mask).sum(1, keepdim=True).float().clamp(min=1.0)
        global_mean = masked.sum(1) / real_len
        first_vec = enc[:, 0, :]

        seq_len, sub_len = x.size(1), x.size(1) // self.num_subsequences
        sub_means = []
        for i in range(self.num_subsequences):
            s, e = i * sub_len, (i + 1) * sub_len if i < self.num_subsequences - 1 else seq_len
            sub_mask = mask[:, s:e]
            sub_enc = masked[:, s:e, :]
            sub_len_real = (~sub_mask).sum(1, keepdim=True).float().clamp(min=1.0)
            sub_means.append(sub_enc.sum(1) / sub_len_real)

        pooled = torch.cat([global_mean, first_vec] + sub_means, 1)
        return self.fc_logits(pooled)

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, vocab_size, max_len, pad_idx=0):
        super().__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim * max_len)
        self.max_len, self.hidden_dim, self.pad_idx = max_len, hidden_dim, pad_idx
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_idx)
        layer = nn.TransformerDecoderLayer(hidden_dim, 8, hidden_dim*4, batch_first=True)
        self.transformer = nn.TransformerDecoder(layer, num_layers=5)
        self.out = nn.Linear(hidden_dim, vocab_size)

    @staticmethod
    def generate_causal_mask(sz, device):
        return torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), 1)

    def forward(self, z, tgt):
        memory = self.fc(z).view(-1, self.max_len, self.hidden_dim)
        tgt_emb = self.embedding(tgt)
        tlen, device = tgt.size(1), tgt.device
        tgt_mask = self.generate_causal_mask(tlen, device)
        tgt_pad_mask = tgt.eq(self.pad_idx)
        dec = self.transformer(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
        return self.out(dec)

class DVAE(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, latent_dim, max_len, rbm_hidden, pad_idx=0):
        super().__init__()
        self.encoder = Encoder(vocab_size, emb_dim, hidden_dim, latent_dim, max_len, pad_idx)
        self.rbm = RBM(latent_dim, rbm_hidden)
        self.decoder = Decoder(latent_dim, hidden_dim, vocab_size, max_len, pad_idx)
        self.latent_dim, self.pad_idx = latent_dim, pad_idx

    def sample_relaxed(self, logits, tau=0.5):
        probs = torch.sigmoid(logits)
        gumbel = -torch.log(-torch.log(torch.rand_like(probs) + 1e-8) + 1e-8)
        return torch.sigmoid((logits + gumbel) / tau), probs

    def sample_hard(self, probs):
        return (probs > 0.5).float()

    def forward(self, enc_in, dec_in, tau=0.5):
        enc_logits = self.encoder(enc_in)
        z_relaxed, probs = self.sample_relaxed(enc_logits, tau)
        z_hard = self.sample_hard(probs)
        dec_out = self.decoder(z_relaxed, dec_in)
        return dec_out, enc_logits, z_relaxed, z_hard


def generate_from_prior(model, tokenizer, device, num_samples=5, gibbs_steps=200, max_gen_len=100):
    model.eval()
    sos, eos, pad = tokenizer.char_to_idx['<SOS>'], tokenizer.char_to_idx['<EOS>'], tokenizer.char_to_idx['<PAD>']
    v = torch.bernoulli(torch.full((num_samples, model.latent_dim), 0.5)).to(device)
    for _ in range(gibbs_steps):
        _, h = model.rbm.sample_h(v)
        _, v = model.rbm.sample_v(h)
    z = v.detach()
    tokens = torch.full((num_samples, 1), sos, dtype=torch.long).to(device)
    finished = torch.zeros(num_samples, dtype=torch.bool).to(device)
    for _ in range(max_gen_len - 1):
        logits = model.decoder(z, tokens)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_tok = torch.multinomial(probs, 1)
        next_tok[finished] = pad
        finished |= next_tok.squeeze() == eos
        tokens = torch.cat([tokens, next_tok], 1)
        if finished.all(): break
    return [tokenizer.decode(tokens[i].cpu().numpy()) for i in range(num_samples)]



def main():
    parser = argparse.ArgumentParser(description="Train DVAE with PCD-30 on QM9")
    parser.add_argument("--dvae_lr", type=float, required=True, help="Learning rate for VAE parts")
    parser.add_argument("--rbm_lr", type=float, required=True, help="Learning rate for RBM parts")
    parser.add_argument("--seed", type=int, required=True, help="Seed for reproducibility")
    args = parser.parse_args()

    # --- 1. Init ---
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Seed {args.seed} on {device}")
    
    # --- 2. Data ---
    download_data()
    print("Loading SMILES...")
    smiles = []
    suppl = Chem.SDMolSupplier('gdb9.sdf', removeHs=False)
    for mol in suppl:
        if mol is not None:
            smiles.append(Chem.MolToSmiles(mol))
    print(f"Loaded {len(smiles)} SMILES")
    
    tokenizer = SMILESTokenizer(smiles)
    max_len = max(len(s) for s in smiles) + 2
    
    train_size = int(0.9 * len(smiles))
    BATCH_SIZE = 128
    
    train_ds = SMILESDataset(smiles[:train_size], tokenizer, max_len)
    val_ds = SMILESDataset(smiles[train_size:], tokenizer, max_len)
    
    # Generator for reproducible shuffle
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, generator=g)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # --- 3. Model & Optimizers ---
    model = DVAE(tokenizer.vocab_size, 128, 256, 128, max_len, 256,
                 pad_idx=tokenizer.char_to_idx['<PAD>']).to(device)
    
    vae_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    opt_vae = torch.optim.Adam(vae_params, lr=args.dvae_lr)
    opt_rbm = torch.optim.Adam(model.rbm.parameters(), lr=args.rbm_lr)
    
    # --- 4. PCD Hyperparams ---
    v_neg = torch.bernoulli(torch.full((BATCH_SIZE, model.latent_dim), 0.5)).to(device)
    CD_STEPS = 30
    EPOCHS = 20
    KLD_MAX = 0.1
    KLD_ANNEAL_EPOCHS = 5
    GRAD_CLIP = 1.0

    metrics = {k: [] for k in ["train_loss", "train_recon", "train_kld", "train_rbm", "val_loss", "val_recon"]}
    best_val_loss = float('inf')

    # --- 5. Training Loop ---
    print(f"Starting Training | Epochs: {EPOCHS} | PCD Steps: {CD_STEPS}")

    for ep in range(EPOCHS):
        model.train()
        tl, tr, tk, tb = [], [], [], []
        
        for i, (enc_in, dec_in, dec_tgt) in enumerate(train_dl):
            # Linear KLD anneal
            kld_w = KLD_MAX * min(1.0, (ep + i / len(train_dl)) / max(1e-8, KLD_ANNEAL_EPOCHS))

            enc_in = enc_in.to(device); dec_in = dec_in.to(device); dec_tgt = dec_tgt.to(device)
            opt_vae.zero_grad(); opt_rbm.zero_grad()

            # Forward
            out, enc_log, z_rel, z_h = model(enc_in, dec_in, tau=0.5)
            
            # Reconstruction loss
            recon = F.cross_entropy(out.reshape(-1, out.size(-1)), dec_tgt.reshape(-1),
                                     ignore_index=tokenizer.char_to_idx['<PAD>'], reduction='mean')

            # KLD
            log_q_per = -F.binary_cross_entropy_with_logits(enc_log, z_h, reduction='none').sum(1)
            log_p_per = -model.rbm.free_energy(z_h) 
            kld_per = (log_q_per - log_p_per)
            kld = kld_per.mean()
            
            vae_loss = recon + kld_w * kld

            # Backward VAE (retain graph for RBM)
            vae_loss.backward(retain_graph=True)

            # Negative phase: PCD
            with torch.no_grad():
                v = v_neg
                for _ in range(CD_STEPS):
                    _, h = model.rbm.sample_h(v)
                    _, v = model.rbm.sample_v(h)
            neg_phase = model.rbm.free_energy(v).mean()

            # Balance Gradients
            (-kld_w * neg_phase).backward()

            # Persist chain
            v_neg = v.detach()

            # Clip and Step
            torch.nn.utils.clip_grad_norm_(vae_params, GRAD_CLIP)
            torch.nn.utils.clip_grad_norm_(model.rbm.parameters(), GRAD_CLIP)
            opt_vae.step(); opt_rbm.step()

            rbm_cd = (model.rbm.free_energy(z_h.detach()) - model.rbm.free_energy(v)).mean().item()
            tl.append(vae_loss.item()); tr.append(recon.item()); tk.append(kld.item()); tb.append(rbm_cd)

            # --- [Feature 1] Log Progress with Gradient Norms every 50 batches ---
            if (i + 1) % 50 == 0:
                gn_vae = grad_norm(vae_params)
                gn_rbm = grad_norm(model.rbm.parameters())
                print(f"Batch {i+1}/{len(train_dl)} | Loss {vae_loss.item():.4f} | Recon {recon.item():.4f} | "
                      f"KLD {kld.item():.4f} | RBM {rbm_cd:.4f} | Grads: VAE {gn_vae:.3f} RBM {gn_rbm:.3f}")

        # Aggregates
        avg_loss, avg_recon, avg_kld, avg_rbm = map(np.mean, [tl, tr, tk, tb])
        metrics["train_loss"].append(avg_loss)
        metrics["train_recon"].append(avg_recon)
        metrics["train_kld"].append(avg_kld)
        metrics["train_rbm"].append(avg_rbm)
        print(f"Epoch {ep+1}/{EPOCHS} | Train Loss {avg_loss:.4f} | Recon {avg_recon:.4f} | KLD {avg_kld:.4f}")

        # Validation
        model.eval(); vl, vr = [], []
        with torch.no_grad():
            for enc_in, dec_in, dec_tgt in val_dl:
                enc_in = enc_in.to(device); dec_in = dec_in.to(device); dec_tgt = dec_tgt.to(device)
                out, enc_log, z_rel, z_h = model(enc_in, dec_in, tau=0.5)
                recon = F.cross_entropy(out.reshape(-1, out.size(-1)), dec_tgt.reshape(-1),
                                         ignore_index=tokenizer.char_to_idx['<PAD>'], reduction='mean')
                log_q_per = -F.binary_cross_entropy_with_logits(enc_log, z_h, reduction='none').sum(1)
                log_p_per = -model.rbm.free_energy(z_h)
                kld_val = (log_q_per - log_p_per).mean()
                vl.append((recon + kld_w * kld_val).item()); vr.append(recon.item())

        v_loss, v_recon = np.mean(vl), np.mean(vr)
        metrics["val_loss"].append(v_loss); metrics["val_recon"].append(v_recon)
        
        # Save Best
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), f'best_model_seed_{args.seed}.pt')
            print(f"  [+] Saved Best Model (Val Loss: {v_loss:.4f})")
            
        # --- [Feature 2] Intermediate Generation every 2 epochs ---
        if (ep + 1) % 2 == 0:
            print(f"--- Epoch {ep+1} Sanity Check ---")
            with torch.no_grad():
                chk_smiles = generate_from_prior(model, tokenizer, device, num_samples=5, gibbs_steps=200, max_gen_len=max_len)
            
            valid_count = 0
            for j, s in enumerate(chk_smiles):
                mol = Chem.MolFromSmiles(s)
                is_valid = mol is not None
                if is_valid: valid_count += 1
                
                # Print string with Checkmark or Cross
                status = "✓" if is_valid else "✗"
                print(f"{j+1}. {s[:60]:<60} {status}")
            
            print(f"Valid: {valid_count}/5")
            print("-------------------------------")

    # --- 6. Post-Training Output ---
    
    # Save Metrics & Plot
    torch.save(metrics, f'metrics_seed_{args.seed}.pt')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(metrics['train_recon'], label='Train Recon')
    ax1.plot(metrics['val_recon'], '--', label='Val Recon')
    ax1.set(title=f'Reconstruction (Seed {args.seed})', xlabel='Epoch', ylabel='Loss')
    ax1.legend(); ax1.grid(True)
    
    ax2_t = ax2.twinx()
    ax2.plot(metrics['train_kld'], color='blue', label='KLD')
    ax2_t.plot(metrics['train_rbm'], color='red', label='RBM CD')
    ax2.set_title(f'Latent Losses (Seed {args.seed})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('KLD', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2_t.set_ylabel('RBM CD', color='red')
    ax2.legend(loc='upper left'); ax2_t.legend(loc='upper right'); ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'training_plot_seed_{args.seed}.png')
    plt.close()

    # --- 7. Final Gen + Eval ---
    print("\nGenerating final samples and calculating metrics...")
    
    with torch.no_grad():
        gen_smiles = generate_from_prior(model, tokenizer, device, num_samples=5000, gibbs_steps=200, max_gen_len=max_len)
    
    with open(f'generated_smiles_seed_{args.seed}.txt', 'w') as f:
        for s in gen_smiles:
            f.write(f"{s}\n")

    valid_mols = []
    valid_smiles = []
    for s in gen_smiles:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            valid_mols.append(m)
            # CHANGE: Canonicalize the Generated output to match Training format
            valid_smiles.append(Chem.MolToSmiles(m))

    # Basic Metrics
    n_gen = len(gen_smiles)
    n_valid = len(valid_smiles)
    validity = n_valid / n_gen * 100 if n_gen > 0 else 0
    
    unique_set = set(valid_smiles)
    n_unique = len(unique_set)
    uniqueness = n_unique / n_valid * 100 if n_valid > 0 else 0
    
    # Create a canonical training set for STRICT string matching
    print("Canonicalizing training set for strict novelty check...")
    train_set_canonical = set()
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m:
            train_set_canonical.add(Chem.MolToSmiles(m))
            
    n_novel = sum(1 for s in unique_set if s not in train_set_canonical)
    novelty = n_novel / n_unique * 100 if n_unique > 0 else 0
    
    # Properties
    logp_vals = [Descriptors.MolLogP(m) for m in valid_mols]
    qed_vals = [QED.qed(m) for m in valid_mols]
    avg_logp = np.mean(logp_vals) if logp_vals else 0
    avg_qed = np.mean(qed_vals) if qed_vals else 0
    
    
    avg_sim, nov_score, per_nov = calculate_tanimoto_stats(valid_smiles, smiles, n_sample=1000)

    # --- PRINT FINAL REPORT ---
    print(f"\n{'='*40}")
    print(f"FINAL RESULTS FOR SEED {args.seed}")
    print(f"{'='*40}")
    print(f"Validity:          {validity:.2f}%")
    print(f"Uniqueness:        {uniqueness:.2f}%")
    print(f"Novelty (Strict):  {novelty:.2f}%")
    print(f"Internal Sim:      {avg_sim:.4f}")
    print(f"Novelty Score:     {nov_score:.4f}")
    print(f"Novel (<0.4):      {per_nov:.1f}%")
    print(f"Avg LogP:          {avg_logp:.4f}")
    print(f"Avg QED:           {avg_qed:.4f}")
    print(f"{'='*40}")
    
    # Save results
    with open(f'final_stats_seed_{args.seed}.txt', 'w') as f:
        f.write(f"Validity: {validity}\nUniqueness: {uniqueness}\nNovelty: {novelty}\n")
        f.write(f"Internal_Similarity: {avg_sim}\n")
        f.write(f"Novelty_Score: {nov_score}\n")
        f.write(f"LogP: {avg_logp}\nQED: {avg_qed}\n")

    print(f"Done. Files saved for Seed {args.seed}.")
    

if __name__ == "__main__":
    main()
