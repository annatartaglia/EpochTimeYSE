'''
    This file contains all functions necessary for training and testing *both* neural networks
'''

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torch.nn import LSTM, Module
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt
from typing import List, Tuple, Sequence, Optional, Callable
import math, csv 
import matplotlib.animation as animation
import os

# ---------------- config for datasets and training ----------------
TIME_SCALE = 200.0
WL_CENTER  = 6500.0
WL_SCALE   = 2000.0
POOLING    = "mean"      

# data truncation control stuff
TRUNC_MIN_LEN = 5         # shortest truncated length to allow (>=1)
TRUNC_INCLUDE_FULL = False # include the full curve as a "truncated" sample too
TRUNC_STRIDE = 2          # subsample truncation lengths (e.g., 2,3,5 to thin out)
MIX_TRUNCATED = True       # if True, mix full + truncated during training
P_TRUNC_MAX = 0.8          # max fraction of truncated samples in a batch
P_TRUNC_GAMMA = 1.5        # ramp exponent for p_trunc across epochs

# optimization stuff
LR = 3e-3
WEIGHT_DECAY = 0.0
BATCH_SIZE_TRAIN = 56
BATCH_SIZE_EVAL  = 8
NUM_EPOCHS = 50
ALPHA_NLL = 5.0            # weight on NLL part of loss

bands = {
    'g': 4900.12, 'r': 6241.27, 'i': 7563.76, 'z': 8690.10,
    'ZTF_r': 6463.75, 'ZTF_g': 4829.50
}

# ---------------- loading ----------------
def load_and_process_data(filenames: Sequence[str]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
        X rows: [scaled_time, log1p(flux), log1p(uncert), scaled_wavelength]
        y:      [tmax_relative, observed_stdev]
    """
    x_all, y_all = [], []
    for f in filenames:
        df = pd.read_csv(f)
        if 'relative_time' not in df.columns:
            continue
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.sort_values('relative_time', kind='mergesort').reset_index(drop=True)

        if 'max_light' not in df or 'stdev' not in df:
            continue
        if not df['max_light'].notna().any() or not df['stdev'].notna().any():
            continue

        max_light_val = df['max_light'].dropna().iloc[0]
        stdev_val     = df['stdev'].dropna().iloc[0]

        t0 = df['relative_time'].min()
        if pd.isna(t0):
            continue

        rows = []
        for _, row in df.iterrows():
            t_raw = row['relative_time']
            if pd.isna(t_raw):
                continue
            t_rel = float(t_raw) - float(t0)
            t_scaled = t_rel / TIME_SCALE

            for band, wl_nm in bands.items():
                fcol = f"{band}_flux"
                ecol = f"{band}_uncert"
                if fcol in df.columns and ecol in df.columns:
                    flux, err = row[fcol], row[ecol]
                    if pd.notna(flux) and pd.notna(err):
                        flux_n   = np.log1p(max(float(flux), 0.0))
                        uncert_n = np.log1p(max(float(err),   0.0))
                        wl_n     = (float(wl_nm) - WL_CENTER) / WL_SCALE
                        rows.append([t_scaled, flux_n, uncert_n, wl_n])

        if not rows:
            continue

        x = np.asarray(rows, dtype=np.float32)
        y = np.asarray([float(max_light_val - t0), float(stdev_val)], dtype=np.float32)
        x_all.append(x); y_all.append(y)

    return x_all, y_all


def load_and_process_data_t0(filenames: Sequence[str]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
        same thing as above but for t0 (trigger phase) network
    """
    x_all, y_all = [], []
    for f in filenames:
        df = pd.read_csv(f)
        if 'relative_time' not in df.columns:
            continue
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.sort_values('relative_time', kind='mergesort').reset_index(drop=True)

        if 'exp_time' not in df or 'exp_stdev' not in df:
            continue
        if not df['exp_time'].notna().any() or not df['exp_stdev'].notna().any():
            continue

        t0_val = df['exp_time'].dropna().iloc[0]
        stdev_val = df['exp_stdev'].dropna().iloc[0]

        t_init = df['relative_time'].min()
        if pd.isna(t_init):
            continue

        rows = []
        for _, row in df.iterrows():
            t_raw = row['relative_time']
            if pd.isna(t_raw):
                continue
            t_rel = float(t_raw) - float(t_init)
            t_scaled = t_rel / TIME_SCALE

            for band, wl_nm in bands.items():
                fcol = f"{band}_flux"
                ecol = f"{band}_uncert"
                if fcol in df.columns and ecol in df.columns:
                    flux, err = row[fcol], row[ecol]
                    if pd.notna(flux) and pd.notna(err):
                        flux_n   = np.log1p(max(float(flux), 0.0))
                        uncert_n = np.log1p(max(float(err),   0.0))
                        wl_n     = (float(wl_nm) - WL_CENTER) / WL_SCALE
                        rows.append([t_scaled, flux_n, uncert_n, wl_n])

        if not rows:
            continue

        x = np.asarray(rows, dtype=np.float32)
        y = np.asarray([float(t0_val - t_init), float(stdev_val)], dtype=np.float32)
        x_all.append(x); y_all.append(y)

    return x_all, y_all

# ---------------- datasets ----------------
class FullCurveDataset(Dataset):
    """
        One item per full curve; pads to global max_T
    """
    def __init__(self, x_data: List[np.ndarray], y_data: List[np.ndarray]):
        self.x = x_data
        self.y = y_data
        self.max_T = max((a.shape[0] for a in x_data), default=0)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        T, F = x.shape
        pad_l = int(self.max_T - T)
        if pad_l > 0:
            pad = np.zeros((pad_l, F), dtype=np.float32)
            pad[:, 0] = -1000.0  # sentinel only in time col
            x = np.vstack([x, pad])
        y = self.y[idx]
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y[0], dtype=torch.float32),
                torch.tensor(y[1], dtype=torch.float32))

class TruncatedCurveDataset(Dataset):
    """
        Dataset of truncated LC segments (cutting each full LC to increasing length from min with given stride)
    """
    def __init__(self,
                 x_data: List[np.ndarray],
                 y_data: List[np.ndarray],
                 min_len: int = TRUNC_MIN_LEN,
                 include_full: bool = TRUNC_INCLUDE_FULL,
                 stride: int = TRUNC_STRIDE):
        self.x_raw = x_data
        self.y_raw = y_data
        self.min_len = max(1, int(min_len))
        self.include_full = bool(include_full)
        self.stride = max(1, int(stride))

        self.index: List[Tuple[int, int]] = []  # (curve_idx, trunc_len)
        self.max_T = 0
        for i, x in enumerate(self.x_raw):
            T = x.shape[0]
            self.max_T = max(self.max_T, T)
            n_max = T if self.include_full else max(self.min_len, T - 1)
            for n in range(self.min_len, n_max + 1, self.stride):
                if 1 <= n <= T:
                    self.index.append((i, n))

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        i, n = self.index[idx]
        x_full = self.x_raw[i]
        y_vec  = self.y_raw[i]
        x = x_full[:n, :]

        T, F = x.shape
        pad_l = int(self.max_T - T)
        if pad_l > 0:
            pad = np.zeros((pad_l, F), dtype=np.float32)
            pad[:, 0] = -1000.0
            x = np.vstack([x, pad])

        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y_vec[0], dtype=torch.float32),
                torch.tensor(y_vec[1], dtype=torch.float32))

# ---------------- model ----------------
class EpochLSTM(Module):
    def __init__(self, input_size=4, h1=64, h2=32, dropout_rate=0.2):
        super().__init__()
        self.lstm1 = LSTM(input_size=input_size, hidden_size=h1, batch_first=True)
        self.lstm2 = LSTM(input_size=h1, hidden_size=h2, batch_first=True)
        self.norm  = nn.LayerNorm(h2)
        self.dropout = nn.Dropout(dropout_rate)

        self.mu_head = nn.Sequential(
            nn.Linear(h2, h2), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h2, 1)
        )
        self.sigma_head = nn.Sequential(
            nn.Linear(h2, h2), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h2, 1)
        )
        self.softplus = nn.Softplus()
        self.mu_bias = nn.Parameter(torch.zeros(1))  # global bias for mu

    def forward(self, x):
        # x: [B,T,4], padded rows are [-1000,0,0,0] in time column
        time_col = x[..., 0]
        lengths = (time_col != -1000).sum(dim=1)  

        keep = lengths > 0
        if not keep.all():
            x = x[keep]; lengths = lengths[keep]
            if x.size(0) == 0:
                return x.new_zeros((0,)), x.new_zeros((0,))

        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        p1, _ = self.lstm1(packed)
        p2, _ = self.lstm2(p1)
        out, _ = pad_packed_sequence(p2, batch_first=True)   

        if POOLING == "mean":
            Bp, Tmax, H = out.shape
            device = out.device
            mask = (torch.arange(Tmax, device=device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1)  
            feats = (out * mask).sum(dim=1) / lengths.unsqueeze(1).clamp(min=1)                          
        else:
            last_idx = (lengths - 1).clamp(min=0)
            feats = out[torch.arange(out.size(0), device=out.device), last_idx]                          

        feats = self.norm(feats)
        feats = self.dropout(feats)

        mu = self.mu_head(feats).squeeze(-1) + self.mu_bias
        sigma_pred = self.softplus(self.sigma_head(feats)).squeeze(-1) + 1e-6  # stdev >= 0
        return mu, sigma_pred
    
# ---------------- loss ----------------
def loss_mean_and_sigma(mu, sigma_pred, y_true, y_obs_sd,
                        alpha=ALPHA_NLL, beta=3.0, gamma_hinge=0.0,
                        sd_floor=1e-3, eps=1e-6):
    """
        alpha: weight on NLL (fit mean with predicted sigma)
        beta : weight on calibration (match predicted sigma to observed sigma)
        gamma_hinge: extra penalty when sigma_pred > y_obs_sd
    """
    y_obs_sd_clamped = torch.clamp(y_obs_sd, min=sd_floor)

    var = sigma_pred**2 + eps
    nll = 0.5 * (torch.log(var) + (y_true - mu)**2 / var).mean()

    log_sigma = torch.log(sigma_pred + eps)
    log_y     = torch.log(y_obs_sd_clamped + eps)
    calib = F.l1_loss(log_sigma, log_y)

    overshoot = torch.relu(sigma_pred - y_obs_sd_clamped)
    hinge = (overshoot**2).mean()

    return alpha * nll + beta * calib + gamma_hinge * hinge

# β warm-up so μ learns first..?
def beta_schedule(epoch, total_epochs, beta_min=0.5, beta_max=3.0):
    t = epoch / max(1, total_epochs-1)
    return beta_min + (beta_max - beta_min) * t

def p_trunc_schedule(epoch: int, total_epochs: int,
                     p_max: float = P_TRUNC_MAX, gamma: float = P_TRUNC_GAMMA) -> float:
    """Fraction of truncated samples to target at epoch (0..E-1)."""
    t = epoch / max(1, total_epochs-1)
    return float(np.clip((t ** gamma) * p_max, 0.0, 1.0))

# ---------------- loaders ----------------
def make_mixed_loader(full_ds: Dataset,
                      trunc_ds: Dataset,
                      batch_size: int,
                      p_trunc: float) -> DataLoader:
    """
        Build a DataLoader sampling a mix of full and truncated items with ratio ~ p_trunc.
    """
    ds = ConcatDataset([full_ds, trunc_ds])
    n_full = len(full_ds)
    n_tr   = len(trunc_ds)
    if n_full == 0 or n_tr == 0:
        raise ValueError("Empty dataset(s).")
    else:
        # weights produce expected proportion ~ p_trunc
        w_full = (1.0 - p_trunc) / max(1, n_full)
        w_tr   = (p_trunc)       / max(1, n_tr)
        weights = [w_full]*n_full + [w_tr]*n_tr
        sampler = WeightedRandomSampler(weights, num_samples=n_full + n_tr, replacement=True)

    return DataLoader(ds, batch_size=batch_size, sampler=sampler)

def class_dataloader(spectral_class, file_dir, spectral_dict, test_t0=False):
    '''
        Make a dataloader of just files corresponding to given spectral class

        spectral_class:: str (ex "SNIa")
        file_dir:: directory containing processed files
        spectral_dict:: dictionary containing YSE dr1 spectral type info (spectral_types.pkl in this repo)
    '''
    names = spectral_dict[spectral_class]
    filenames = [
        f"{file_dir}lc_{name}_processed.csv"
        for name in names
        if os.path.exists(f"{file_dir}lc_{name}_processed.csv")
    ]    
    if test_t0:
        all_xdata, all_ydata = load_and_process_data_t0(filenames)
    else:
        all_xdata, all_ydata = load_and_process_data(filenames)
    if len(all_xdata) == 0:
        raise ValueError("no data.")
    
    N = len(all_ydata)
    data = TruncatedCurveDataset(all_xdata, all_ydata,
                                  min_len=1, include_full=True, stride=1)
    return DataLoader(data, batch_size=30, shuffle=False), N


# ---------------- training ----------------
def train_and_eval(model: Module,
                   full_train_ds: Dataset,
                   trunc_train_ds: Optional[Dataset],
                   val_loader: DataLoader,
                   optimizer: optim.Optimizer,
                   num_epochs: int = NUM_EPOCHS,
                   alpha: float = ALPHA_NLL):
    """
        If MIX_TRUNCATED is True and trunc_train_ds is provided, rebuild a mixed train loader
        each epoch with rising p_trunc! Otherwise train on full_train_ds only.
    """
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        beta = beta_schedule(epoch, total_epochs=num_epochs)

        if MIX_TRUNCATED and trunc_train_ds is not None and len(trunc_train_ds) > 0:
            p_trunc = p_trunc_schedule(epoch, num_epochs)
            train_loader = make_mixed_loader(full_train_ds, trunc_train_ds, BATCH_SIZE_TRAIN, p_trunc)
        else:
            train_loader = DataLoader(full_train_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
            p_trunc = 0.0

        running, nb = 0.0, 0
        for xb, yb, ysd in train_loader:
            xb, yb, ysd = xb.float(), yb.float(), ysd.float()
            optimizer.zero_grad()
            mu, sigma_pred = model(xb)
            if mu.size(0) != yb.size(0):
                keep = (xb[...,0] != -1000).sum(dim=1) > 0
                yb = yb[keep]; ysd = ysd[keep]
                mu = mu[keep]; sigma_pred = sigma_pred[keep]
            loss = loss_mean_and_sigma(mu, sigma_pred, yb, ysd, alpha=alpha, beta=beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.item()); nb += 1

        train_losses.append(running / max(1, nb))

        # ---- eval
        model.eval()
        ev, nb = 0.0, 0
        with torch.no_grad():
            for xb, yb, ysd in val_loader:
                mu, sigma_pred = model(xb.float())
                if mu.size(0) != yb.size(0):
                    keep = (xb[...,0] != -1000).sum(dim=1) > 0
                    yb = yb[keep]; ysd = ysd[keep]
                    mu = mu[keep]; sigma_pred = sigma_pred[keep]
                ev += float(loss_mean_and_sigma(mu, sigma_pred, yb.float(), ysd.float(), alpha=alpha, beta=beta))
                nb += 1
        val_losses.append(ev / max(1, nb))

        # sanity check: μ should not be constant forever
        with torch.no_grad():
            xb_dbg, yb_dbg, ysd_dbg = next(iter(val_loader))
            mu_dbg, s_dbg = model(xb_dbg.float())
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"train {train_losses[-1]:.4f} | val {val_losses[-1]:.4f} "
                  f"| beta={beta:.2f} | p_trunc={p_trunc:.2f} "
                  f"| mu.std={float(mu_dbg.std()):.2f} | sigma.med={float(s_dbg.median()):.2f}")

    return train_losses, val_losses

# ---------------- plotting ----------------
def plot_test(model, test_loader, num_plots=10):
    model.eval()
    plot_count = 0
    x, y, y_stdev = next(iter(test_loader))
    with torch.no_grad():
            x = x.float()
            mu, sigma_pred = model(x)
            pred_mean = mu.cpu().numpy()
            pred_std  = sigma_pred.cpu().numpy()
            B = x.size(0)
            for j in range(B):
                if plot_count >= num_plots: return
                mask = (x[j, :, 0] != -1000)
                if mask.sum().item() == 0: continue
                sample = x[j][mask].cpu().numpy()
                times = sample[:, 0] * TIME_SCALE
                fluxes = sample[:, 1]
                errors = sample[:, 2]
                wavelengths = sample[:, 3] * WL_SCALE + WL_CENTER
                pm = pred_mean[j]; ps = pred_std[j]
                true_tmax  = y[j].item()
                true_stdev = y_stdev[j].item()

                fig, ax = plt.subplots()
                ax.set_title(f"Sample {plot_count}")
                ax.set_xlabel("Relative Time")
                ax.set_ylabel("Flux (log1p)")
                unique_wl = np.unique(wavelengths)
                cmap = plt.cm.tab10.colors
                cmap_map = {wl: cmap[i % len(cmap)] for i, wl in enumerate(unique_wl)}
                for t, f, e, wl in zip(times, fluxes, errors, wavelengths):
                    ax.errorbar(t, f, yerr=e, fmt='o', alpha=0.6, color=cmap_map.get(wl, 'gray'))
                ax.axvline(true_tmax, color='black', linestyle='--', label='True $t_{0}$')
                ax.axvspan(true_tmax - true_stdev, true_tmax + true_stdev, color='black', alpha=0.2)
                ax.axvline(pm, color='red', label='Pred $t_{0}$')
                ax.axvspan(pm - ps, pm + ps, color='red', alpha=0.2)
                ax.legend(); plt.tight_layout(); plt.show()
                plot_count += 1

def nll_vs_phase_binned(model, val_trunc_loader,
                        bin_width=5.0,
                        phase_min=-200.0, phase_max=200.0,
                        out_csv="nll_vs_phase_binned.csv",
                        sd_floor=1e-6):
    """
    traditional Gaussian NLL in fixed bins of phase = (last_observed_time_raw - true_tmax)
    Writes per-bin: center, mean_nll, std_nll, count.

    Set (phase_min, phase_max) to reasonable window for data.
    """
    assert phase_max > phase_min, "phase_max must be > phase_min"
    nbins = int(math.ceil((phase_max - phase_min) / bin_width))
    edges0 = phase_min
    inv_bw = 1.0 / bin_width

    counts = [0] * nbins
    sums   = [0.0] * nbins
    sumsq  = [0.0] * nbins

    const = 0.5 * math.log(2.0 * math.pi)
    model.eval()
    dev = next(model.parameters()).device

    with torch.no_grad():
        for xb, yb, _ in val_trunc_loader:
            # last observed time in *raw* units
            mask = (xb[..., 0] != -1000)
            lengths = mask.sum(dim=1).clamp(min=1)
            last_idx = lengths - 1
            last_time_raw = xb[torch.arange(xb.size(0)), last_idx, 0] * TIME_SCALE  

            # move to device for model
            xb_d = xb.to(dev, non_blocking=True).float()
            yb_d = yb.to(dev, non_blocking=True).float()

            mu, sigma_pred = model(xb_d)
            sigma = torch.clamp(sigma_pred, min=sd_floor)

            # NLL = 0.5*log(2π) + log(σ) + 0.5*((y - μ)^2 / σ^2)
            nll = (const + torch.log(sigma) + 0.5 * ((yb_d - mu)**2) / (sigma**2)).cpu()
            phase = (last_time_raw - yb).cpu()

            # bins
            for p, v in zip(phase.tolist(), nll.tolist()):
                b = int(math.floor((p - edges0) * inv_bw))
                if 0 <= b < nbins:
                    counts[b] += 1
                    sums[b]   += v
                    sumsq[b]  += v*v

            del xb, yb, xb_d, yb_d, mu, sigma_pred, sigma, nll, phase, last_time_raw, mask, lengths, last_idx

    # write CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["center_phase", "mean_nll", "std_nll", "count"])
        for i in range(nbins):
            c = counts[i]
            center = edges0 + (i + 0.5) * bin_width
            if c == 0:
                w.writerow([center, "", "", 0])
            else:
                mean = sums[i] / c
                var  = max((sumsq[i] / c) - (mean * mean), 0.0)
                std  = math.sqrt(var) if c > 1 else 0.0
                w.writerow([center, mean, std, c])

    print(f"Wrote {out_csv} with {nbins} bins over [{phase_min}, {phase_max}] (bin width {bin_width}).")

# ---------------- animation: NN over N truncations of ONE processed LC ----------------
def animate_nn_over_truncations(model,
                                x_full: np.ndarray,   # [t_scaled, log1p(flux), log1p(uncert), wl_scaled]
                                y_true: float = None, # true tmax, optional
                                y_sd: float = None,   # true stdev, optional
                                fname: str = "lc_nn_animation.mp4",
                                fps: int = 5,
                                dpi: int = 150):
    """
        Make MP4 animation for a single processed light curve sequence.
    """
    assert x_full.ndim == 2 and x_full.shape[1] == 4, "x_full must be [T,4]"
    T = x_full.shape[0]
    if T < 1:
        raise ValueError("Empty sequence")

    # Precompute predictions for all truncations
    model.eval()
    device = next(model.parameters()).device
    mus, sigmas = [], []
    with torch.no_grad():
        for k in range(1, T+1):
            xk = torch.tensor(x_full[:k], dtype=torch.float32, device=device).unsqueeze(0)  # [1,k,4]
            mu, sigma = model(xk)
            mus.append(float(mu.squeeze().detach().cpu()))
            sigmas.append(float(sigma.squeeze().detach().cpu()))
    mus    = np.asarray(mus)
    sigmas = np.asarray(sigmas)

    # Plot data from processed sequence
    times_raw = x_full[:, 0] * TIME_SCALE
    flux_log  = x_full[:, 1]
    wl_raw    = x_full[:, 3] * WL_SCALE + WL_CENTER

    unique_wl = np.unique(wl_raw)
    cmap = plt.cm.tab10.colors
    wl_to_color = {wl: cmap[i % len(cmap)] for i, wl in enumerate(unique_wl)}

    xmin, xmax = float(times_raw.min()), float(times_raw.max())
    ymin = float(np.nanmin(flux_log)) if flux_log.size else 0.0
    ymax = float(np.nanmax(flux_log)) if flux_log.size else 1.0
    if ymin == ymax:
        ymin -= 1.0; ymax += 1.0

    fig, ax = plt.subplots()
    ax.set_xlim(xmin, xmax+5)
    ax.set_ylim(ymin-.5, ymax+.5)
    ax.set_xlabel("Relative Time (raw units)")
    ax.set_ylabel("Flux (log1p)")
    ax.set_title(fname)

    # True tmax (optional)
    if y_true is not None:
        ax.axvline(y_true, color='black', linestyle='--', alpha=0.9, linewidth=1.2, zorder=0, label='True $t_{max}$')
        if y_sd is not None and np.isfinite(y_sd):
            ax.axvspan(y_true - y_sd, y_true + y_sd, color='gray', alpha=0.2, label='±1σ true')

    point_handles = {wl: ax.plot([], [], 'o', color=wl_to_color[wl], alpha=0.6, markersize=5)[0]
                     for wl in unique_wl}

    # Predicted line & band
    pred_line = ax.axvline(mus[0], color='red', linewidth=1.0, label='NN $t_{max}$')
    pred_band = ax.axvspan(mus[0] - sigmas[0], mus[0] + sigmas[0], color='red', alpha=0.18, label='±1σ (NN)')

    ax.legend(loc='upper left')

    def _update_band(span, x0, x1):
        span.remove()
        return ax.axvspan(x0, x1, color='red', alpha=0.18)

    def init():
        nonlocal pred_band
        for wl in unique_wl:
            point_handles[wl].set_data([], [])
        pred_line.set_xdata([mus[0]])
        pred_band = _update_band(pred_band, mus[0] - sigmas[0], mus[0] + sigmas[0])
        return list(point_handles.values()) + [pred_line, pred_band]

    def update(frame_idx):
        k = frame_idx + 1  
        tr_k = times_raw[:k]
        fl_k = flux_log[:k]
        wl_k = wl_raw[:k]
        for wl in unique_wl:
            sel = (wl_k == wl)
            point_handles[wl].set_data(tr_k[sel], fl_k[sel])
        pred_line.set_xdata([mus[k-1]])
        nonlocal pred_band
        pred_band = _update_band(pred_band, mus[k-1] - sigmas[k-1], mus[k-1] + sigmas[k-1])
        return list(point_handles.values()) + [pred_line, pred_band]

    ani = animation.FuncAnimation(fig, update, frames=T, init_func=init, blit=False, interval=200, repeat=False)
    ani.save(fname, writer='ffmpeg', fps=fps, dpi=dpi)
    plt.close(fig)