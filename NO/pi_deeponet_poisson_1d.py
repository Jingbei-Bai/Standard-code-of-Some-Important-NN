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


def sample_force_coeffs(n_funcs, k_max=6, seed=0):
    rng = np.random.default_rng(seed)
    amp = 1.0 / np.arange(1, k_max + 1, dtype=np.float32)
    coeff = rng.normal(0.0, 1.0, size=(n_funcs, k_max)).astype(np.float32) * amp
    return coeff


def forcing_from_coeffs_np(coeff, x):

    k = np.arange(1, coeff.shape[1] + 1, dtype=np.float32)[None, None, :]
    x_e = x[..., None]
    val = np.sum(coeff[:, None, :] * np.sin(np.pi * k * x_e), axis=-1)
    return val


def solution_from_coeffs_np(coeff, x):

    k = np.arange(1, coeff.shape[1] + 1, dtype=np.float32)[None, None, :]
    x_e = x[..., None]
    denom = (np.pi * k) ** 2
    val = np.sum(coeff[:, None, :] * np.sin(np.pi * k * x_e) / denom, axis=-1)
    return val


def forcing_from_coeffs_torch(coeff, x):

    k = torch.arange(
        1,
        coeff.shape[1] + 1,
        device=coeff.device,
        dtype=coeff.dtype,
    ).view(1, 1, -1)
    x_e = x.unsqueeze(-1)
    return torch.sum(coeff.unsqueeze(1) * torch.sin(np.pi * k * x_e), dim=-1)


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
            branch_i = np.repeat(f_sensor[i : i + 1], nx, axis=0)
            branch_t = torch.tensor(branch_i, dtype=torch.float32, device=device)
            pred = model(branch_t, x_eval_t).cpu().numpy().squeeze()
            u_pred[i, :] = pred

    rel_l2 = np.linalg.norm(u_pred - u_ref) / (np.linalg.norm(u_ref) + 1e-12)
    return u_pred, u_ref, f_sensor, rel_l2


def train_pi_deeponet_poisson_1d(
    n_train_funcs=1000,
    n_test_funcs=200,
    n_sensor=100,
    k_max=6,
    hidden=128,
    latent_dim=128,
    n_layers=3,
    batch_funcs=32,
    n_col_per_func=64,
    adam_epochs=3000,
    lbfgs_iters=300,
    lr=1e-3,
    lambda_bc=50.0,
    out_dir=None,
    device=None,
):
    if device is None:
        device = set_device()
    if out_dir is None:
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
    os.makedirs(out_dir, exist_ok=True)

    sensor_x = np.linspace(0.0, 1.0, n_sensor, dtype=np.float32)
    coeff_train_np = sample_force_coeffs(n_train_funcs, k_max=k_max, seed=11)
    coeff_test_np = sample_force_coeffs(n_test_funcs, k_max=k_max, seed=22)
    f_sensor_train_np = forcing_from_coeffs_np(coeff_train_np, sensor_x[None, :]).astype(np.float32)

    coeff_train = torch.tensor(coeff_train_np, dtype=torch.float32, device=device)
    f_sensor_train = torch.tensor(f_sensor_train_np, dtype=torch.float32, device=device)

    model = DeepONet(
        branch_in=n_sensor,
        trunk_in=1,
        width=hidden,
        latent_dim=latent_dim,
        n_layers=n_layers,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    mse_pde_history = []
    mse_bc_history = []

    for epoch in range(1, adam_epochs + 1):
        model.train()
        func_idx = torch.randint(0, n_train_funcs, (batch_funcs,), device=device)
        coeff_batch = coeff_train[func_idx]
        branch_batch = f_sensor_train[func_idx]


        x_col = torch.rand((batch_funcs, n_col_per_func), dtype=torch.float32, device=device)
        branch_rep = branch_batch.unsqueeze(1).repeat(1, n_col_per_func, 1).reshape(-1, n_sensor)
        x_flat = x_col.reshape(-1, 1).clone().detach().requires_grad_(True)
        u_pred = model(branch_rep, x_flat)
        u_x = torch.autograd.grad(
            u_pred, x_flat, grad_outputs=torch.ones_like(u_pred), create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x_flat, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0]

        f_col = forcing_from_coeffs_torch(coeff_batch, x_col).reshape(-1, 1)
        r = -u_xx - f_col
        mse_pde = torch.mean(r ** 2)


        x0 = torch.zeros((batch_funcs, 1), dtype=torch.float32, device=device)
        x1 = torch.ones((batch_funcs, 1), dtype=torch.float32, device=device)
        u0 = model(branch_batch, x0)
        u1 = model(branch_batch, x1)
        mse_bc = 0.5 * (torch.mean(u0 ** 2) + torch.mean(u1 ** 2))

        loss = mse_pde + lambda_bc * mse_bc
        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_history.append(loss.item())
        mse_pde_history.append(mse_pde.item())
        mse_bc_history.append(mse_bc.item())

        if epoch == 1 or epoch % 300 == 0:
            print(
                f"adam epoch {epoch}/{adam_epochs}, loss={loss.item():.6e}, "
                f"mse_pde={mse_pde.item():.6e}, mse_bc={mse_bc.item():.6e}"
            )

    if lbfgs_iters > 0:

        n_lb_funcs = min(256, n_train_funcs)
        func_idx = torch.randperm(n_train_funcs, device=device)[:n_lb_funcs]
        coeff_batch = coeff_train[func_idx]
        branch_batch = f_sensor_train[func_idx]
        n_col_lb = n_col_per_func
        x_col = torch.rand((n_lb_funcs, n_col_lb), dtype=torch.float32, device=device)

        opt_lbfgs = optim.LBFGS(
            model.parameters(),
            max_iter=lbfgs_iters,
            tolerance_grad=1e-10,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt_lbfgs.zero_grad()
            branch_rep = branch_batch.unsqueeze(1).repeat(1, n_col_lb, 1).reshape(-1, n_sensor)
            x_flat = x_col.reshape(-1, 1).clone().detach().requires_grad_(True)
            u_pred = model(branch_rep, x_flat)
            u_x = torch.autograd.grad(
                u_pred, x_flat, grad_outputs=torch.ones_like(u_pred), create_graph=True
            )[0]
            u_xx = torch.autograd.grad(
                u_x, x_flat, grad_outputs=torch.ones_like(u_x), create_graph=True
            )[0]
            f_col = forcing_from_coeffs_torch(coeff_batch, x_col).reshape(-1, 1)
            r = -u_xx - f_col
            mse_pde = torch.mean(r ** 2)

            x0 = torch.zeros((n_lb_funcs, 1), dtype=torch.float32, device=device)
            x1 = torch.ones((n_lb_funcs, 1), dtype=torch.float32, device=device)
            u0 = model(branch_batch, x0)
            u1 = model(branch_batch, x1)
            mse_bc = 0.5 * (torch.mean(u0 ** 2) + torch.mean(u1 ** 2))
            loss_val = mse_pde + lambda_bc * mse_bc
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

    idx = 0
    plt.figure(figsize=(12, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(sensor_x, f_sensor_test[idx], "k-", lw=2, label="input forcing f(x)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Random Test Forcing")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x_eval, u_ref[idx], "k-", lw=2, label="exact")
    plt.plot(x_eval, u_pred[idx], "r--", lw=2, label="PI-DeepONet")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("Poisson Solution Prediction")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "pi_deeponet_poisson_example.png")
    plt.savefig(fig_path, dpi=180)
    print("saved figure to", fig_path)

    plt.figure(figsize=(7, 4.5))
    plt.plot(loss_history, label="total", lw=1.5)
    plt.plot(mse_pde_history, label="mse_pde", lw=1.2)
    plt.plot(np.array(mse_bc_history) * lambda_bc, label="lambda_bc*mse_bc", lw=1.2)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss terms")
    plt.title("PI-DeepONet Training History")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    loss_fig = os.path.join(out_dir, "pi_deeponet_poisson_loss.png")
    plt.savefig(loss_fig, dpi=180)
    print("saved loss figure to", loss_fig)
    return rel_l2


if __name__ == "__main__":
        device = set_device()
        print("device:", device)
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
        train_pi_deeponet_poisson_1d(out_dir=out_dir, device=device)



