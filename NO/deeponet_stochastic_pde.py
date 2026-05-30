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
    def __init__(self, branch_in, trunk_in=1, width=128, latent_dim=128, n_layers=3):
        super().__init__()
        self.branch = build_mlp(branch_in, width, latent_dim, n_layers=n_layers)
        self.trunk = build_mlp(trunk_in, width, latent_dim, n_layers=n_layers)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, branch_input, trunk_input):
        b = self.branch(branch_input)
        t = self.trunk(trunk_input)
        return torch.sum(b * t, dim=1, keepdim=True) + self.bias


def sample_force_coeffs(n_samples, k_max=5, seed=0):
    rng = np.random.default_rng(seed)
    amp = 1.0 / np.arange(1, k_max + 1, dtype=np.float32)
    coeff = rng.normal(0.0, 1.0, size=(n_samples, k_max)).astype(np.float32) * amp
    return coeff


def forcing_np(coeff, x):

    k = np.arange(1, coeff.shape[1] + 1, dtype=np.float32)[None, None, :]
    x_e = x[..., None]
    return np.sum(coeff[:, None, :] * np.sin(np.pi * k * x_e), axis=-1)


def solution_np(coeff, x, xi):

    kappa = 1.0 + 0.5 * xi
    k = np.arange(1, coeff.shape[1] + 1, dtype=np.float32)[None, None, :]
    x_e = x[..., None]
    denom = kappa[:, None, None] * ((np.pi * k) ** 2)
    return np.sum(coeff[:, None, :] * np.sin(np.pi * k * x_e) / denom, axis=-1)


def build_dataset(coeff, xi, sensor_x, n_query_per_sample=64, seed=0):
    rng = np.random.default_rng(seed)
    n_samples = coeff.shape[0]

    f_sensor = forcing_np(coeff, sensor_x[None, :]).astype(np.float32)
    branch_base = np.concatenate([f_sensor, xi[:, None].astype(np.float32)], axis=1)

    x_query = rng.uniform(0.0, 1.0, size=(n_samples, n_query_per_sample)).astype(np.float32)
    u_query = solution_np(coeff, x_query, xi).astype(np.float32)

    branch = np.repeat(branch_base, n_query_per_sample, axis=0).astype(np.float32)
    trunk = x_query.reshape(-1, 1).astype(np.float32)
    y = u_query.reshape(-1, 1).astype(np.float32)
    return branch, trunk, y, branch_base


def evaluate_operator(model, coeff_test, xi_test, sensor_x, x_eval, device):
    model.eval()
    n_test = coeff_test.shape[0]
    nx = x_eval.shape[0]

    f_sensor = forcing_np(coeff_test, sensor_x[None, :]).astype(np.float32)
    branch_base = np.concatenate([f_sensor, xi_test[:, None].astype(np.float32)], axis=1)
    u_ref = solution_np(
        coeff_test,
        np.repeat(x_eval[None, :], n_test, axis=0).astype(np.float32),
        xi_test,
    ).astype(np.float32)
    u_pred = np.zeros_like(u_ref, dtype=np.float32)

    x_eval_t = torch.tensor(x_eval[:, None], dtype=torch.float32, device=device)
    with torch.no_grad():
        for i in range(n_test):
            b_i = np.repeat(branch_base[i : i + 1], nx, axis=0).astype(np.float32)
            b_t = torch.tensor(b_i, dtype=torch.float32, device=device)
            pred = model(b_t, x_eval_t).cpu().numpy().squeeze()
            u_pred[i] = pred

    rel_l2 = np.linalg.norm(u_pred - u_ref) / (np.linalg.norm(u_ref) + 1e-12)
    return u_pred, u_ref, branch_base, rel_l2


def train_deeponet_stochastic_pde(
    n_train_samples=1400,
    n_test_samples=300,
    n_sensor=100,
    k_max=5,
    n_query_per_sample=96,
    hidden=128,
    latent_dim=128,
    n_layers=3,
    batch_size=4096,
    adam_epochs=3200,
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

    sensor_x = np.linspace(0.0, 1.0, n_sensor, dtype=np.float32)
    coeff_train = sample_force_coeffs(n_train_samples, k_max=k_max, seed=1111)
    coeff_test = sample_force_coeffs(n_test_samples, k_max=k_max, seed=2222)
    xi_train = np.random.uniform(-1.0, 1.0, size=(n_train_samples,)).astype(np.float32)
    xi_test = np.random.uniform(-1.0, 1.0, size=(n_test_samples,)).astype(np.float32)

    branch_np, trunk_np, y_np, _ = build_dataset(
        coeff_train,
        xi_train,
        sensor_x=sensor_x,
        n_query_per_sample=n_query_per_sample,
        seed=3333,
    )
    branch_t = torch.tensor(branch_np, dtype=torch.float32, device=device)
    trunk_t = torch.tensor(trunk_np, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_np, dtype=torch.float32, device=device)
    n_samples = y_t.shape[0]

    model = DeepONet(
        branch_in=n_sensor + 1,
        trunk_in=1,
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

    x_eval = np.linspace(0.0, 1.0, 201, dtype=np.float32)
    u_pred, u_ref, branch_test, rel_l2 = evaluate_operator(
        model=model,
        coeff_test=coeff_test,
        xi_test=xi_test,
        sensor_x=sensor_x,
        x_eval=x_eval,
        device=device,
    )
    print(f"test relative L2 error={rel_l2:.6e}")

    idx = 0
    plt.figure(figsize=(12, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(sensor_x, branch_test[idx, :-1], "k-", lw=2, label=f"forcing, xi={branch_test[idx,-1]:.2f}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Random Stochastic Input")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x_eval, u_ref[idx], "k-", lw=2, label="exact")
    plt.plot(x_eval, u_pred[idx], "r--", lw=2, label="DeepONet")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("Stochastic PDE Solution")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "deeponet_stochastic_pde_example.png")
    plt.savefig(fig_path, dpi=180)
    print("saved figure to", fig_path)

    plt.figure(figsize=(7, 4.5))
    plt.plot(loss_history, lw=1.5)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("train MSE")
    plt.title("DeepONet Stochastic PDE Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    loss_fig = os.path.join(out_dir, "deeponet_stochastic_pde_loss.png")
    plt.savefig(loss_fig, dpi=180)
    print("saved loss figure to", loss_fig)
    return rel_l2


if __name__ == "__main__":
        device = set_device()
        print("device:", device)
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
        train_deeponet_stochastic_pde(out_dir=out_dir, device=device)



