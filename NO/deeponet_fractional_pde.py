import math
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


def build_mlp(in_dim, hidden, out_dim, n_layers=3):
    layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
    for _ in range(n_layers - 1):
        layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
    layers.append(nn.Linear(hidden, out_dim))
    net = nn.Sequential(*layers)
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    return net


class DeepONet(nn.Module):
    def __init__(self, branch_in, trunk_in=2, width=128, latent_dim=128, n_layers=3):
        super().__init__()
        self.branch = build_mlp(branch_in, width, latent_dim, n_layers=n_layers)
        self.trunk = build_mlp(trunk_in, width, latent_dim, n_layers=n_layers)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, branch_input, trunk_input):
        b = self.branch(branch_input)
        t = self.trunk(trunk_input)
        return torch.sum(b * t, dim=1, keepdim=True) + self.bias


def sample_coeffs(n_funcs, k_max=4, seed=0):
    rng = np.random.default_rng(seed)
    amp = 1.0 / np.arange(1, k_max + 1, dtype=np.float32)
    coeff = rng.normal(0.0, 1.0, size=(n_funcs, k_max)).astype(np.float32) * amp
    return coeff


def u_fractional_np(coeff, t, x, beta):

    k = np.arange(1, coeff.shape[1] + 1, dtype=np.float32)[None, None, :]
    sin_part = np.sin(np.pi * k * x[..., None])
    base = np.sum(coeff[:, None, :] * sin_part, axis=-1)
    return (t ** beta) * base


def source_fractional_np(coeff, t, x, alpha, beta):

    c_ab = math.gamma(beta + 1.0) / math.gamma(beta + 1.0 - alpha)
    k = np.arange(1, coeff.shape[1] + 1, dtype=np.float32)[None, None, :]
    sin_part = np.sin(np.pi * k * x[..., None])
    term_t = c_ab * (t[..., None] ** (beta - alpha))
    term_x = (np.pi * k) ** 2 * (t[..., None] ** beta)
    val = np.sum(coeff[:, None, :] * (term_t + term_x) * sin_part, axis=-1)
    return val


def build_dataset(coeff, sensor_x, n_query_per_func, alpha, beta, seed=0):
    rng = np.random.default_rng(seed)
    n_funcs = coeff.shape[0]


    t1 = np.ones((n_funcs, sensor_x.shape[0]), dtype=np.float32)
    x_sensor = np.repeat(sensor_x[None, :], n_funcs, axis=0).astype(np.float32)
    f_sensor = source_fractional_np(coeff, t1, x_sensor, alpha=alpha, beta=beta).astype(np.float32)


    t_query = rng.uniform(0.0, 1.0, size=(n_funcs, n_query_per_func)).astype(np.float32)
    x_query = rng.uniform(0.0, 1.0, size=(n_funcs, n_query_per_func)).astype(np.float32)
    u_query = u_fractional_np(coeff, t_query, x_query, beta=beta).astype(np.float32)

    branch = np.repeat(f_sensor, n_query_per_func, axis=0).astype(np.float32)
    trunk = np.stack([t_query.reshape(-1), x_query.reshape(-1)], axis=1).astype(np.float32)
    y = u_query.reshape(-1, 1).astype(np.float32)
    return branch, trunk, y, f_sensor


def evaluate_operator(model, coeff_test, sensor_x, t_eval, x_eval, alpha, beta, device):
    model.eval()
    n_test = coeff_test.shape[0]
    nt = t_eval.shape[0]
    nx = x_eval.shape[0]

    t_mesh, x_mesh = np.meshgrid(t_eval, x_eval, indexing="ij")
    t_row = np.repeat(t_mesh.reshape(1, -1), n_test, axis=0).astype(np.float32)
    x_row = np.repeat(x_mesh.reshape(1, -1), n_test, axis=0).astype(np.float32)
    u_ref = u_fractional_np(coeff_test, t_row, x_row, beta=beta).astype(np.float32).reshape(n_test, nt, nx)

    t1 = np.ones((n_test, sensor_x.shape[0]), dtype=np.float32)
    x_sensor = np.repeat(sensor_x[None, :], n_test, axis=0).astype(np.float32)
    f_sensor = source_fractional_np(coeff_test, t1, x_sensor, alpha=alpha, beta=beta).astype(np.float32)

    tx_grid = np.stack([t_mesh.reshape(-1), x_mesh.reshape(-1)], axis=1).astype(np.float32)
    tx_t = torch.tensor(tx_grid, dtype=torch.float32, device=device)
    u_pred = np.zeros_like(u_ref, dtype=np.float32)

    with torch.no_grad():
        for i in range(n_test):
            b_i = np.repeat(f_sensor[i : i + 1], tx_grid.shape[0], axis=0).astype(np.float32)
            b_t = torch.tensor(b_i, dtype=torch.float32, device=device)
            pred = model(b_t, tx_t).cpu().numpy().reshape(nt, nx)
            u_pred[i] = pred

    rel_l2 = np.linalg.norm(u_pred - u_ref) / (np.linalg.norm(u_ref) + 1e-12)
    return u_pred, u_ref, f_sensor, rel_l2


def train_deeponet_fractional_pde(
    alpha=0.6,
    beta=2.0,
    n_train_funcs=1200,
    n_test_funcs=200,
    n_sensor=100,
    k_max=4,
    n_query_per_func=96,
    hidden=128,
    latent_dim=128,
    n_layers=3,
    batch_size=4096,
    adam_epochs=3000,
    lbfgs_iters=300,
    lr=1e-3,
    out_dir=None,
    device=None,
):
    if device is None:
        device = set_device()
    if out_dir is None:
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
    os.makedirs(out_dir, exist_ok=True)

    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha should be in (0,1)")
    if beta <= alpha:
        raise ValueError("beta should be greater than alpha")

    sensor_x = np.linspace(0.0, 1.0, n_sensor, dtype=np.float32)
    coeff_train = sample_coeffs(n_train_funcs, k_max=k_max, seed=101)
    coeff_test = sample_coeffs(n_test_funcs, k_max=k_max, seed=202)

    branch_np, trunk_np, y_np, _ = build_dataset(
        coeff_train, sensor_x=sensor_x, n_query_per_func=n_query_per_func, alpha=alpha, beta=beta, seed=303
    )

    branch_t = torch.tensor(branch_np, dtype=torch.float32, device=device)
    trunk_t = torch.tensor(trunk_np, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_np, dtype=torch.float32, device=device)
    n_samples = y_t.shape[0]

    model = DeepONet(
        branch_in=n_sensor,
        trunk_in=2,
        width=hidden,
        latent_dim=latent_dim,
        n_layers=n_layers,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    for epoch in range(1, adam_epochs + 1):
        model.train()
        perm = torch.randperm(n_samples, device=device)
        n_batches = int(np.ceil(n_samples / batch_size))
        epoch_loss = 0.0
        for bidx in range(n_batches):
            idx = perm[bidx * batch_size : (bidx + 1) * batch_size]
            pred = model(branch_t[idx], trunk_t[idx])
            loss = torch.mean((pred - y_t[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        epoch_loss /= max(1, n_batches)
        loss_history.append(epoch_loss)
        if epoch == 1 or epoch % 300 == 0:
            print(f"adam epoch {epoch}/{adam_epochs}, mse={epoch_loss:.6e}")

    if lbfgs_iters > 0:
        opt_lbfgs = optim.LBFGS(
            model.parameters(),
            max_iter=lbfgs_iters,
            tolerance_grad=1e-10,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt_lbfgs.zero_grad()
            pred = model(branch_t, trunk_t)
            loss_val = torch.mean((pred - y_t) ** 2)
            loss_val.backward()
            return loss_val

        print("starting L-BFGS stage")
        opt_lbfgs.step(closure)

    t_eval = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    x_eval = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    u_pred, u_ref, f_sensor_test, rel_l2 = evaluate_operator(
        model, coeff_test=coeff_test, sensor_x=sensor_x, t_eval=t_eval, x_eval=x_eval, alpha=alpha, beta=beta, device=device
    )
    print(f"test relative L2 error={rel_l2:.6e}")

    idx = 0
    TT, XX = np.meshgrid(t_eval, x_eval, indexing="ij")
    plt.figure(figsize=(12, 4.5))
    plt.subplot(1, 2, 1)
    plt.title("Reference u(t,x)")
    pcm1 = plt.pcolormesh(TT, XX, u_ref[idx], shading="auto", cmap="RdYlBu")
    plt.colorbar(pcm1)
    plt.xlabel("t")
    plt.ylabel("x")

    plt.subplot(1, 2, 2)
    plt.title("DeepONet prediction u(t,x)")
    pcm2 = plt.pcolormesh(TT, XX, u_pred[idx], shading="auto", cmap="RdYlBu")
    plt.colorbar(pcm2)
    plt.xlabel("t")
    plt.ylabel("x")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "deeponet_fractional_pde_example.png")
    plt.savefig(fig_path, dpi=180)
    print("saved figure to", fig_path)

    plt.figure(figsize=(7, 4.5))
    plt.plot(loss_history, lw=1.5)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("train MSE")
    plt.title("DeepONet Fractional PDE Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    loss_fig = os.path.join(out_dir, "deeponet_fractional_pde_loss.png")
    plt.savefig(loss_fig, dpi=180)
    print("saved loss figure to", loss_fig)
    return rel_l2


if __name__ == "__main__":
        device = set_device()
        print("device:", device)
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
        train_deeponet_fractional_pde(out_dir=out_dir, device=device)



