import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


torch.manual_seed(0)
np.random.seed(0)


def set_device():
    return torch.device("cuda")


def sample_coeffs(n_samples, k_max=5, seed=0):
    rng = np.random.default_rng(seed)
    amp = 1.0 / np.arange(1, k_max + 1, dtype=np.float32)
    coeff = rng.normal(0.0, 1.0, size=(n_samples, k_max)).astype(np.float32) * amp
    return coeff


def forcing_and_solution_from_coeffs(coeff, x_grid):

    k = np.arange(1, coeff.shape[1] + 1, dtype=np.float32)[:, None]
    s = np.sin(np.pi * k * x_grid[None, :]).astype(np.float32)
    f = coeff @ s
    u = (coeff / ((np.pi * np.arange(1, coeff.shape[1] + 1, dtype=np.float32)) ** 2)) @ s
    return f.astype(np.float32), u.astype(np.float32)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x):

        bsz, _, n = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        n_freq = x_ft.shape[-1]
        m = min(self.modes, n_freq)
        out_ft = torch.zeros(
            bsz, self.out_channels, n_freq, dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :m] = torch.einsum("bim,iom->bom", x_ft[:, :, :m], self.weight[:, :, :m])
        x_out = torch.fft.irfft(out_ft, n=n, dim=-1)
        return x_out


class FNO1d(nn.Module):
    def __init__(self, modes=16, width=64, n_layers=4):
        super().__init__()
        self.modes = modes
        self.width = width
        self.fc0 = nn.Linear(2, width)
        self.spec_layers = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(n_layers)]
        )
        self.w_layers = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, f_in):

        bsz, n = f_in.shape
        x_grid = torch.linspace(0.0, 1.0, n, device=f_in.device).view(1, n, 1).repeat(bsz, 1, 1)
        x = torch.cat([f_in.unsqueeze(-1), x_grid], dim=-1)
        x = self.fc0(x).permute(0, 2, 1)

        for spec, w in zip(self.spec_layers, self.w_layers):
            x = torch.nn.functional.gelu(spec(x) + w(x))

        x = x.permute(0, 2, 1)
        x = torch.nn.functional.gelu(self.fc1(x))
        x = self.fc2(x).squeeze(-1)


        grid = x_grid[:, :, 0]
        return grid * (1.0 - grid) * x


def evaluate_model(model, f_test, u_test, device):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(f_test, dtype=torch.float32, device=device)).cpu().numpy()
    rel_l2 = np.linalg.norm(pred - u_test) / (np.linalg.norm(u_test) + 1e-12)
    return pred, rel_l2


def train_fno_poisson_1d(
    n_train=1200,
    n_test=200,
    n_grid=256,
    k_max=5,
    modes=16,
    width=64,
    n_layers=4,
    batch_size=64,
    adam_epochs=2500,
    lr=1e-3,
    weight_decay=1e-6,
    out_dir=None,
    device=None,
):
    if device is None:
        device = set_device()
    if out_dir is None:
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
    os.makedirs(out_dir, exist_ok=True)

    x_grid = np.linspace(0.0, 1.0, n_grid, dtype=np.float32)
    c_train = sample_coeffs(n_train, k_max=k_max, seed=11)
    c_test = sample_coeffs(n_test, k_max=k_max, seed=22)
    f_train, u_train = forcing_and_solution_from_coeffs(c_train, x_grid)
    f_test, u_test = forcing_and_solution_from_coeffs(c_test, x_grid)


    f_mean = f_train.mean(axis=0, keepdims=True)
    f_std = f_train.std(axis=0, keepdims=True) + 1e-6
    u_mean = u_train.mean(axis=0, keepdims=True)
    u_std = u_train.std(axis=0, keepdims=True) + 1e-6

    f_train_n = (f_train - f_mean) / f_std
    f_test_n = (f_test - f_mean) / f_std
    u_train_n = (u_train - u_mean) / u_std

    model = FNO1d(modes=modes, width=width, n_layers=n_layers).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    f_train_t = torch.tensor(f_train_n, dtype=torch.float32, device=device)
    u_train_t = torch.tensor(u_train_n, dtype=torch.float32, device=device)
    n_batches = int(np.ceil(n_train / batch_size))

    loss_history = []
    for epoch in range(1, adam_epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        for bidx in range(n_batches):
            idx = perm[bidx * batch_size : (bidx + 1) * batch_size]
            pred = model(f_train_t[idx])
            loss = torch.mean((pred - u_train_t[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        epoch_loss /= max(1, n_batches)
        loss_history.append(epoch_loss)
        if epoch == 1 or epoch % 250 == 0:
            print(f"adam epoch {epoch}/{adam_epochs}, mse={epoch_loss:.6e}")

    pred_n, rel_l2_n = evaluate_model(model, f_test_n, (u_test - u_mean) / u_std, device=device)
    pred = pred_n * u_std + u_mean
    rel_l2 = np.linalg.norm(pred - u_test) / (np.linalg.norm(u_test) + 1e-12)
    print(f"test relative L2 error={rel_l2:.6e}")

    idx = 0
    plt.figure(figsize=(12, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(x_grid, f_test[idx], "k-", lw=2, label="forcing f")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Random Test Forcing")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x_grid, u_test[idx], "k-", lw=2, label="exact u")
    plt.plot(x_grid, pred[idx], "r--", lw=2, label="FNO")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("Poisson Solution")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "fno_poisson_1d_example.png")
    plt.savefig(fig_path, dpi=180)
    print("saved figure to", fig_path)

    plt.figure(figsize=(7, 4.5))
    plt.plot(loss_history, lw=1.5)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("train MSE")
    plt.title("FNO 1D Poisson Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    loss_fig = os.path.join(out_dir, "fno_poisson_1d_loss.png")
    plt.savefig(loss_fig, dpi=180)
    print("saved loss figure to", loss_fig)
    return rel_l2, rel_l2_n


if __name__ == "__main__":
        device = set_device()
        print("device:", device)
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
        train_fno_poisson_1d(out_dir=out_dir, device=device)



