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


def sample_pitch_signal_coeffs(n_samples, n_modes=5, seed=0):
    rng = np.random.default_rng(seed)
    a0 = rng.uniform(-4.0, 4.0, size=(n_samples, 1)).astype(np.float32)
    amp_scale = 1.0 / np.arange(1, n_modes + 1, dtype=np.float32)
    a_sin = (rng.normal(0.0, 2.0, size=(n_samples, n_modes)).astype(np.float32)) * amp_scale
    a_cos = (rng.normal(0.0, 2.0, size=(n_samples, n_modes)).astype(np.float32)) * amp_scale
    return a0, a_sin, a_cos


def eval_alpha_deg(a0, a_sin, a_cos, t, t_end):

    n_modes = a_sin.shape[1]
    k = np.arange(1, n_modes + 1, dtype=np.float32)[None, None, :]
    tau = t / t_end
    tau_e = tau[..., None]
    val = a0[:, None, :] + np.sum(
        a_sin[:, None, :] * np.sin(2.0 * np.pi * k * tau_e)
        + a_cos[:, None, :] * np.cos(2.0 * np.pi * k * tau_e),
        axis=-1,
        keepdims=True,
    )
    return val.squeeze(-1)


def simulate_pitching_airfoil_cl(alpha_deg, t, tau_lag=0.08, c_rate=0.35):

    t = np.asarray(t).reshape(-1)
    dt = float(t[1] - t[0])
    alpha_rad = np.deg2rad(alpha_deg)
    dalpha = np.gradient(alpha_rad, dt, axis=1)

    n_samples, nt = alpha_rad.shape
    z = np.zeros((n_samples, nt), dtype=np.float32)
    for n in range(nt - 1):
        z[:, n + 1] = z[:, n] + dt * (alpha_rad[:, n] - z[:, n]) / tau_lag
    cl = 2.0 * np.pi * (z + c_rate * dalpha)
    return cl.astype(np.float32)


def build_dataset(a0, a_sin, a_cos, sensor_t, query_t, t_end, tau_lag=0.08, c_rate=0.35):
    n_samples = a0.shape[0]
    nt_sensor = sensor_t.shape[0]
    nt_query = query_t.shape[0]

    t_sensor_row = np.repeat(sensor_t[None, :], n_samples, axis=0).astype(np.float32)
    alpha_sensor = eval_alpha_deg(a0, a_sin, a_cos, t_sensor_row, t_end=t_end).astype(np.float32)

    t_query_row = np.repeat(query_t[None, :], n_samples, axis=0).astype(np.float32)
    alpha_query = eval_alpha_deg(a0, a_sin, a_cos, t_query_row, t_end=t_end).astype(np.float32)
    cl_query = simulate_pitching_airfoil_cl(alpha_query, query_t, tau_lag=tau_lag, c_rate=c_rate)

    branch = np.repeat(alpha_sensor, nt_query, axis=0).astype(np.float32)
    trunk = np.tile(query_t[:, None], (n_samples, 1)).astype(np.float32)
    y = cl_query.reshape(-1, 1).astype(np.float32)
    return branch, trunk, y, alpha_sensor, cl_query


def evaluate_operator(model, a0, a_sin, a_cos, sensor_t, t_eval, t_end, device, tau_lag=0.08, c_rate=0.35):
    model.eval()
    n_test = a0.shape[0]
    nt = t_eval.shape[0]

    t_sensor_row = np.repeat(sensor_t[None, :], n_test, axis=0).astype(np.float32)
    alpha_sensor = eval_alpha_deg(a0, a_sin, a_cos, t_sensor_row, t_end=t_end).astype(np.float32)

    t_eval_row = np.repeat(t_eval[None, :], n_test, axis=0).astype(np.float32)
    alpha_eval = eval_alpha_deg(a0, a_sin, a_cos, t_eval_row, t_end=t_end).astype(np.float32)
    cl_ref = simulate_pitching_airfoil_cl(alpha_eval, t_eval, tau_lag=tau_lag, c_rate=c_rate).astype(np.float32)

    cl_pred = np.zeros_like(cl_ref, dtype=np.float32)
    t_eval_t = torch.tensor(t_eval[:, None], dtype=torch.float32, device=device)
    with torch.no_grad():
        for i in range(n_test):
            b_i = np.repeat(alpha_sensor[i : i + 1], nt, axis=0).astype(np.float32)
            b_t = torch.tensor(b_i, dtype=torch.float32, device=device)
            pred = model(b_t, t_eval_t).cpu().numpy().squeeze()
            cl_pred[i] = pred

    rel_l2 = np.linalg.norm(cl_pred - cl_ref) / (np.linalg.norm(cl_ref) + 1e-12)
    return cl_pred, cl_ref, alpha_sensor, rel_l2


def train_deeponet_pitching_airfoil(
    n_train_cases=1600,
    n_test_cases=300,
    n_sensor=120,
    t_end=2.0,
    n_modes=5,
    hidden=128,
    latent_dim=128,
    n_layers=3,
    batch_size=4096,
    adam_epochs=3500,
    lbfgs_iters=300,
    lr=1e-3,
    tau_lag=0.08,
    c_rate=0.35,
    out_dir=None,
    device=None,
):
    if device is None:
        device = set_device()
    if out_dir is None:
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
    os.makedirs(out_dir, exist_ok=True)

    sensor_t = np.linspace(0.0, t_end, n_sensor, dtype=np.float32)
    query_t = sensor_t.copy()

    a0_train, a_sin_train, a_cos_train = sample_pitch_signal_coeffs(n_train_cases, n_modes=n_modes, seed=41)
    a0_test, a_sin_test, a_cos_test = sample_pitch_signal_coeffs(n_test_cases, n_modes=n_modes, seed=42)

    branch_np, trunk_np, y_np, _, _ = build_dataset(
        a0_train,
        a_sin_train,
        a_cos_train,
        sensor_t=sensor_t,
        query_t=query_t,
        t_end=t_end,
        tau_lag=tau_lag,
        c_rate=c_rate,
    )
    branch_t = torch.tensor(branch_np, dtype=torch.float32, device=device)
    trunk_t = torch.tensor(trunk_np, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_np, dtype=torch.float32, device=device)
    n_samples = y_t.shape[0]

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

    t_eval = np.linspace(0.0, t_end, n_sensor, dtype=np.float32)
    cl_pred, cl_ref, alpha_sensor_test, rel_l2 = evaluate_operator(
        model=model,
        a0=a0_test,
        a_sin=a_sin_test,
        a_cos=a_cos_test,
        sensor_t=sensor_t,
        t_eval=t_eval,
        t_end=t_end,
        device=device,
        tau_lag=tau_lag,
        c_rate=c_rate,
    )
    print(f"test relative L2 error={rel_l2:.6e}")

    idx = 0
    plt.figure(figsize=(12, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(sensor_t, alpha_sensor_test[idx], "k-", lw=2, label="pitch angle alpha(t) [deg]")
    plt.xlabel("t")
    plt.ylabel("alpha (deg)")
    plt.title("Random Pitching Motion")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t_eval, cl_ref[idx], "k-", lw=2, label="reference Cl")
    plt.plot(t_eval, cl_pred[idx], "r--", lw=2, label="DeepONet Cl")
    plt.xlabel("t")
    plt.ylabel("Cl")
    plt.title("Pitching Airfoil Lift Prediction")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "deeponet_pitching_airfoil_example.png")
    plt.savefig(fig_path, dpi=180)
    print("saved figure to", fig_path)

    plt.figure(figsize=(7, 4.5))
    plt.plot(loss_history, lw=1.5)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("train MSE")
    plt.title("DeepONet Pitching Airfoil Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    loss_fig = os.path.join(out_dir, "deeponet_pitching_airfoil_loss.png")
    plt.savefig(loss_fig, dpi=180)
    print("saved loss figure to", loss_fig)
    return rel_l2


if __name__ == "__main__":
        device = set_device()
        print("device:", device)
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
        train_deeponet_pitching_airfoil(out_dir=out_dir, device=device)



