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


def sample_fourier_coeffs(n_funcs, k_max=5, seed=0):
    rng = np.random.default_rng(seed)
    c0 = rng.uniform(-1.0, 1.0, size=(n_funcs, 1)).astype(np.float32)
    amps = 1.0 / np.arange(1, k_max + 1, dtype=np.float32)
    a = rng.normal(0.0, 0.8, size=(n_funcs, k_max)).astype(np.float32) * amps
    b = rng.normal(0.0, 0.8, size=(n_funcs, k_max)).astype(np.float32) * amps
    return c0, a, b


def eval_input_func(c0, a, b, x):

    k = np.arange(1, a.shape[1] + 1, dtype=np.float32)[None, None, :]
    x_e = x[..., None]
    sin_part = np.sin(np.pi * k * x_e)
    cos_part = np.cos(np.pi * k * x_e)
    val = c0[:, None, :] + np.sum(a[:, None, :] * sin_part + b[:, None, :] * cos_part, axis=-1, keepdims=True)
    return val.squeeze(-1)


def eval_antiderivative(c0, a, b, x):

    k = np.arange(1, a.shape[1] + 1, dtype=np.float32)[None, None, :]
    denom = np.pi * k
    x_e = x[..., None]
    term_sin = a[:, None, :] * (1.0 - np.cos(np.pi * k * x_e)) / denom
    term_cos = b[:, None, :] * np.sin(np.pi * k * x_e) / denom
    val = c0[:, None, :] * x[..., None] + np.sum(term_sin + term_cos, axis=-1, keepdims=True)
    return val.squeeze(-1)


def build_supervised_dataset(c0, a, b, sensor_x, n_query_per_func=64, seed=0):
    n_funcs = c0.shape[0]
    rng = np.random.default_rng(seed)

    sensor_x_row = sensor_x[None, :]
    a_sensor = eval_input_func(c0, a, b, sensor_x_row).astype(np.float32)

    x_query = rng.uniform(0.0, 1.0, size=(n_funcs, n_query_per_func)).astype(np.float32)
    u_query = eval_antiderivative(c0, a, b, x_query).astype(np.float32)

    branch = np.repeat(a_sensor, n_query_per_func, axis=0).astype(np.float32)
    trunk = x_query.reshape(-1, 1).astype(np.float32)
    y = u_query.reshape(-1, 1).astype(np.float32)
    return branch, trunk, y, a_sensor


def evaluate_operator(model, c0, a, b, sensor_x, x_eval, device):
    model.eval()
    n_funcs = c0.shape[0]
    nx = x_eval.shape[0]
    sensor_x_row = sensor_x[None, :]
    a_sensor = eval_input_func(c0, a, b, sensor_x_row).astype(np.float32)

    u_ref = eval_antiderivative(
        c0,
        a,
        b,
        np.repeat(x_eval[None, :], n_funcs, axis=0).astype(np.float32),
    ).astype(np.float32)
    u_pred = np.zeros_like(u_ref, dtype=np.float32)

    with torch.no_grad():
        x_eval_t = torch.tensor(x_eval[:, None], dtype=torch.float32, device=device)
        for i in range(n_funcs):
            branch_i = np.repeat(a_sensor[i : i + 1], nx, axis=0)
            branch_t = torch.tensor(branch_i, dtype=torch.float32, device=device)
            pred = model(branch_t, x_eval_t).cpu().numpy().squeeze()
            u_pred[i, :] = pred

    rel_l2 = np.linalg.norm(u_pred - u_ref) / (np.linalg.norm(u_ref) + 1e-12)
    return u_pred, u_ref, rel_l2, a_sensor


def train_deeponet_antiderivative(
    n_train_funcs=800,
    n_test_funcs=200,
    n_sensor=100,
    k_max=5,
    n_query_per_func=64,
    hidden=128,
    latent_dim=128,
    n_layers=3,
    batch_size=2048,
    adam_epochs=2500,
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
    c0_train, a_train, b_train = sample_fourier_coeffs(n_train_funcs, k_max=k_max, seed=1)
    c0_test, a_test, b_test = sample_fourier_coeffs(n_test_funcs, k_max=k_max, seed=2)

    train_branch, train_trunk, train_y, _ = build_supervised_dataset(
        c0_train, a_train, b_train, sensor_x, n_query_per_func=n_query_per_func, seed=3
    )
    n_samples = train_y.shape[0]

    branch_t = torch.tensor(train_branch, dtype=torch.float32, device=device)
    trunk_t = torch.tensor(train_trunk, dtype=torch.float32, device=device)
    y_t = torch.tensor(train_y, dtype=torch.float32, device=device)

    model = DeepONet(
        branch_in=n_sensor,
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
        epoch_loss = 0.0
        n_batches = int(np.ceil(n_samples / batch_size))
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
        if epoch == 1 or epoch % 250 == 0:
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
    u_pred, u_ref, rel_l2, a_sensor_test = evaluate_operator(
        model, c0_test, a_test, b_test, sensor_x, x_eval, device=device
    )
    print(f"test relative L2 error={rel_l2:.6e}")

    idx = 0
    plt.figure(figsize=(12, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(sensor_x, a_sensor_test[idx], "k-", lw=2, label="input function a(x)")
    plt.xlabel("x")
    plt.ylabel("a(x)")
    plt.title("Random Test Input Function")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x_eval, u_ref[idx], "k-", lw=2, label="exact integral")
    plt.plot(x_eval, u_pred[idx], "r--", lw=2, label="DeepONet")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("Antiderivative Prediction")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "deeponet_antiderivative_example.png")
    plt.savefig(fig_path, dpi=180)
    print("saved figure to", fig_path)

    plt.figure(figsize=(7, 4.5))
    plt.plot(loss_history, lw=1.6)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("train MSE")
    plt.title("DeepONet Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    loss_fig = os.path.join(out_dir, "deeponet_antiderivative_loss.png")
    plt.savefig(loss_fig, dpi=180)
    print("saved loss figure to", loss_fig)
    return rel_l2


if __name__ == "__main__":
        device = set_device()
        print("device:", device)
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
        train_deeponet_antiderivative(out_dir=out_dir, device=device)



