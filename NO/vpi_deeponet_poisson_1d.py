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


class VariationalPIDeepONet(nn.Module):


    def __init__(self, branch_in, trunk_in=1, width=128, latent_dim=128, n_layers=3):
        super().__init__()
        self.core = DeepONet(
            branch_in=branch_in,
            trunk_in=trunk_in,
            width=width,
            latent_dim=latent_dim,
            n_layers=n_layers,
        )

    def forward(self, branch_input, trunk_input):
        x = trunk_input[:, 0:1]
        raw = self.core(branch_input, trunk_input)
        return x * (1.0 - x) * raw


def sample_force_coeffs(n_funcs, k_max=6, seed=0):
    rng = np.random.default_rng(seed)
    amp = 1.0 / np.arange(1, k_max + 1, dtype=np.float32)
    coeff = rng.normal(0.0, 1.0, size=(n_funcs, k_max)).astype(np.float32) * amp
    return coeff


def forcing_from_coeffs_np(coeff, x):
    k = np.arange(1, coeff.shape[1] + 1, dtype=np.float32)[None, None, :]
    x_e = x[..., None]
    return np.sum(coeff[:, None, :] * np.sin(np.pi * k * x_e), axis=-1)


def solution_from_coeffs_np(coeff, x):
    k = np.arange(1, coeff.shape[1] + 1, dtype=np.float32)[None, None, :]
    x_e = x[..., None]
    denom = (np.pi * k) ** 2
    return np.sum(coeff[:, None, :] * np.sin(np.pi * k * x_e) / denom, axis=-1)


def forcing_from_coeffs_torch(coeff, x):

    k = torch.arange(1, coeff.shape[1] + 1, dtype=coeff.dtype, device=coeff.device).view(1, 1, -1)
    x_e = x.unsqueeze(-1)
    return torch.sum(coeff.unsqueeze(1) * torch.sin(np.pi * k * x_e), dim=-1)


def gauss_legendre_01(n_quad):
    xi, wi = np.polynomial.legendre.leggauss(n_quad)
    x = 0.5 * (xi + 1.0)
    w = 0.5 * wi
    return x.astype(np.float32), w.astype(np.float32)


def build_test_basis(n_test, xq_np):

    k = np.arange(1, n_test + 1, dtype=np.float32)[:, None]
    x_row = xq_np[None, :]
    v = np.sin(np.pi * k * x_row).astype(np.float32)
    dv = (np.pi * k * np.cos(np.pi * k * x_row)).astype(np.float32)
    return v, dv


def variational_residual_matrix(model, branch_batch, coeff_batch, xq_t, wq_t, v_t, dv_t):

    bsz = branch_batch.shape[0]
    nq = xq_t.shape[0]

    branch_rep = branch_batch.unsqueeze(1).expand(bsz, nq, branch_batch.shape[1]).reshape(-1, branch_batch.shape[1])
    x_rep = xq_t.view(1, nq, 1).expand(bsz, nq, 1).reshape(-1, 1).clone().detach().requires_grad_(True)

    u = model(branch_rep, x_rep)
    u_x = torch.autograd.grad(u, x_rep, grad_outputs=torch.ones_like(u), create_graph=True)[0].reshape(bsz, nq)

    x_bq = xq_t.view(1, nq).expand(bsz, nq)
    f_bq = forcing_from_coeffs_torch(coeff_batch, x_bq)

    wu = u_x * wq_t.view(1, nq)
    wf = f_bq * wq_t.view(1, nq)
    lhs = torch.einsum("bq,mq->bm", wu, dv_t)
    rhs = torch.einsum("bq,mq->bm", wf, v_t)
    return lhs - rhs


def evaluate_operator(model, coeff_test, sensor_x, x_eval, device):
    model.eval()
    n_test = coeff_test.shape[0]
    nx = x_eval.shape[0]

    f_sensor = forcing_from_coeffs_np(coeff_test, sensor_x[None, :]).astype(np.float32)
    u_ref = solution_from_coeffs_np(
        coeff_test,
        np.repeat(x_eval[None, :], n_test, axis=0).astype(np.float32),
    ).astype(np.float32)
    u_pred = np.zeros_like(u_ref, dtype=np.float32)

    x_eval_t = torch.tensor(x_eval[:, None], dtype=torch.float32, device=device)
    with torch.no_grad():
        for i in range(n_test):
            branch_i = np.repeat(f_sensor[i : i + 1], nx, axis=0).astype(np.float32)
            branch_t = torch.tensor(branch_i, dtype=torch.float32, device=device)
            pred = model(branch_t, x_eval_t).cpu().numpy().squeeze()
            u_pred[i, :] = pred

    rel_l2 = np.linalg.norm(u_pred - u_ref) / (np.linalg.norm(u_ref) + 1e-12)
    return u_pred, u_ref, f_sensor, rel_l2


def train_vpi_deeponet_poisson_1d(
    n_train_funcs=1200,
    n_test_funcs=200,
    n_sensor=100,
    k_max=6,
    hidden=128,
    latent_dim=128,
    n_layers=3,
    n_test_basis=8,
    n_quad=80,
    batch_funcs=64,
    adam_epochs=5000,
    lbfgs_iters=400,
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
    coeff_train_np = sample_force_coeffs(n_train_funcs, k_max=k_max, seed=111)
    coeff_test_np = sample_force_coeffs(n_test_funcs, k_max=k_max, seed=222)
    f_sensor_train_np = forcing_from_coeffs_np(coeff_train_np, sensor_x[None, :]).astype(np.float32)

    coeff_train = torch.tensor(coeff_train_np, dtype=torch.float32, device=device)
    f_sensor_train = torch.tensor(f_sensor_train_np, dtype=torch.float32, device=device)

    xq_np, wq_np = gauss_legendre_01(n_quad=n_quad)
    v_np, dv_np = build_test_basis(n_test=n_test_basis, xq_np=xq_np)
    xq_t = torch.tensor(xq_np, dtype=torch.float32, device=device)
    wq_t = torch.tensor(wq_np, dtype=torch.float32, device=device)
    v_t = torch.tensor(v_np, dtype=torch.float32, device=device)
    dv_t = torch.tensor(dv_np, dtype=torch.float32, device=device)

    model = VariationalPIDeepONet(
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
        idx = torch.randint(0, n_train_funcs, (batch_funcs,), device=device)
        coeff_batch = coeff_train[idx]
        branch_batch = f_sensor_train[idx]

        R = variational_residual_matrix(
            model=model,
            branch_batch=branch_batch,
            coeff_batch=coeff_batch,
            xq_t=xq_t,
            wq_t=wq_t,
            v_t=v_t,
            dv_t=dv_t,
        )
        loss = torch.mean(R ** 2)

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_history.append(loss.item())
        if epoch == 1 or epoch % 300 == 0:
            print(f"adam epoch {epoch}/{adam_epochs}, variational_mse={loss.item():.6e}")

    if lbfgs_iters > 0:
        n_lb_funcs = min(256, n_train_funcs)
        idx = torch.randperm(n_train_funcs, device=device)[:n_lb_funcs]
        coeff_batch = coeff_train[idx]
        branch_batch = f_sensor_train[idx]

        opt_lbfgs = optim.LBFGS(
            model.parameters(),
            max_iter=lbfgs_iters,
            tolerance_grad=1e-10,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt_lbfgs.zero_grad()
            R = variational_residual_matrix(
                model=model,
                branch_batch=branch_batch,
                coeff_batch=coeff_batch,
                xq_t=xq_t,
                wq_t=wq_t,
                v_t=v_t,
                dv_t=dv_t,
            )
            loss_val = torch.mean(R ** 2)
            loss_val.backward()
            return loss_val

        print("starting L-BFGS stage")
        opt_lbfgs.step(closure)

    x_eval = np.linspace(0.0, 1.0, 201, dtype=np.float32)
    u_pred, u_ref, f_sensor_test, rel_l2 = evaluate_operator(
        model=model,
        coeff_test=coeff_test_np,
        sensor_x=sensor_x,
        x_eval=x_eval,
        device=device,
    )
    print(f"test relative L2 error={rel_l2:.6e}")

    idx_show = 0
    plt.figure(figsize=(12, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(sensor_x, f_sensor_test[idx_show], "k-", lw=2, label="forcing f(x)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Random Test Forcing")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x_eval, u_ref[idx_show], "k-", lw=2, label="exact")
    plt.plot(x_eval, u_pred[idx_show], "r--", lw=2, label="Variational PI-DeepONet")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("Poisson Solution")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "vpi_deeponet_poisson_example.png")
    plt.savefig(fig_path, dpi=180)
    print("saved figure to", fig_path)

    plt.figure(figsize=(7, 4.5))
    plt.plot(loss_history, lw=1.6)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("variational residual MSE")
    plt.title("Variational PI-DeepONet Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    loss_fig = os.path.join(out_dir, "vpi_deeponet_poisson_loss.png")
    plt.savefig(loss_fig, dpi=180)
    print("saved loss figure to", loss_fig)
    return rel_l2


if __name__ == "__main__":
        device = set_device()
        print("device:", device)
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
        train_vpi_deeponet_poisson_1d(out_dir=out_dir, device=device)



