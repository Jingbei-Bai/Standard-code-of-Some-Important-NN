import argparse
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


class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=1, hidden=128, n_layers=4):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, tx):
        return self.net(tx)


class AdaptiveLossWeights(nn.Module):


    def __init__(self):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(3, dtype=torch.float32))

    def weights(self):
        w = torch.nn.functional.softplus(self.raw) + 1e-6
        return 3.0 * w / torch.sum(w)

    def adaptive_term(self, mse_f, mse_ic, mse_bc):
        w = self.weights()
        return w[0] * mse_f + w[1] * mse_ic + w[2] * mse_bc

    def lambdas(self):
        with torch.no_grad():
            return self.weights().detach().cpu().numpy()


def initial_condition(x):
    return (x ** 2) * torch.cos(np.pi * x)


def sampler(n_f=20000, n_ic=200, n_bc=100, T=1.0, x_lb=-1.0, x_rb=1.0):
    t_f = np.random.rand(n_f, 1) * T
    x_f = np.random.rand(n_f, 1) * (x_rb - x_lb) + x_lb
    X_f = np.hstack([t_f, x_f]).astype(np.float32)

    x_ic = np.random.rand(n_ic, 1) * (x_rb - x_lb) + x_lb
    t_ic = np.zeros((n_ic, 1), dtype=np.float32)
    X_ic = np.hstack([t_ic, x_ic]).astype(np.float32)
    u_ic = (x_ic ** 2) * np.cos(np.pi * x_ic)
    u_ic = u_ic.astype(np.float32)

    t_bc = np.random.rand(n_bc, 1) * T
    x_left = np.ones((n_bc, 1), dtype=np.float32) * x_lb
    x_right = np.ones((n_bc, 1), dtype=np.float32) * x_rb
    X_bc = np.vstack([np.hstack([t_bc, x_left]), np.hstack([t_bc, x_right])]).astype(np.float32)
    u_bc = -np.ones((2 * n_bc, 1), dtype=np.float32)

    return X_f, X_ic, u_ic, X_bc, u_bc


def pde_residual(model, tx, D):
    tx = tx.clone().detach().requires_grad_(True)
    u = model(tx)
    grads = torch.autograd.grad(
        u,
        tx,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]
    u_t = grads[:, 0:1]
    u_x = grads[:, 1:2]
    u_xx = torch.autograd.grad(
        u_x,
        tx,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
    )[0][:, 1:2]
    return u_t - D * u_xx + 5.0 * (u ** 3 - u)


def solve_reference(D, T, Nx, Nt, x_lb=-1.0, x_rb=1.0):
    x = np.linspace(x_lb, x_rb, Nx)
    dx = x[1] - x[0]
    dt = T / (Nt - 1)
    r = D * dt / (dx * dx)

    u = (x ** 2) * np.cos(np.pi * x)
    u_left = -1.0
    u_right = -1.0

    U = np.zeros((Nx, Nt), dtype=np.float64)
    U[:, 0] = u

    n_int = Nx - 2
    if n_int <= 0:
        raise ValueError("Nx too small")

    a_main = (1.0 + r) * np.ones(n_int)
    a_off = -0.5 * r * np.ones(n_int - 1)
    b_main = (1.0 - r) * np.ones(n_int)
    b_off = 0.5 * r * np.ones(n_int - 1)
    A = np.diag(a_main) + np.diag(a_off, -1) + np.diag(a_off, 1)
    B = np.diag(b_main) + np.diag(b_off, -1) + np.diag(b_off, 1)

    for n in range(0, Nt - 1):
        Bu = B.dot(u[1:-1])
        rhs = Bu - dt * 5.0 * (u[1:-1] ** 3 - u[1:-1])
        rhs[0] += 0.5 * r * u_left
        rhs[-1] += 0.5 * r * u_right

        u_int = np.linalg.solve(A, rhs)
        u_np1 = np.zeros_like(u)
        u_np1[0] = u_left
        u_np1[-1] = u_right
        u_np1[1:-1] = u_int
        u = u_np1
        U[:, n + 1] = u

    U[0, :] = u_left
    U[-1, :] = u_right
    return U


def train_alancahn_self_adaptive_pinn(
    device=None,
    out_dir=None,
    D=1e-4,
    T=1.0,
    x_lb=-1.0,
    x_rb=1.0,
    n_f=20000,
    n_ic=300,
    n_bc=300,
    stage1_epochs=12000,
    stage2_lbfgs_iters=1200,
    lr_model=1e-3,
    lr_lambda=2e-3,
    beta_adaptive=1.0,
):
    if device is None:
        device = set_device()
    if out_dir is None:
        out_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "ode_solvers_outputs", "pinns")
        )
    os.makedirs(out_dir, exist_ok=True)

    model = MLP(in_dim=2, out_dim=1, hidden=128, n_layers=4).to(device)
    adaptive = AdaptiveLossWeights().to(device)

    optimizer_model = optim.Adam(model.parameters(), lr=lr_model)
    optimizer_lambda = optim.Adam(adaptive.parameters(), lr=lr_lambda)

    lambda_history = []
    loss_history = []
    mse_f_history = []
    mse_ic_history = []
    mse_bc_history = []

    for epoch in range(1, stage1_epochs + 1):
        model.train()
        X_f, X_ic, u_ic, X_bc, u_bc = sampler(
            n_f=n_f, n_ic=n_ic, n_bc=n_bc, T=T, x_lb=x_lb, x_rb=x_rb
        )

        X_f_t = torch.tensor(X_f, dtype=torch.float32, device=device)
        X_ic_t = torch.tensor(X_ic, dtype=torch.float32, device=device)
        u_ic_t = torch.tensor(u_ic, dtype=torch.float32, device=device)
        X_bc_t = torch.tensor(X_bc, dtype=torch.float32, device=device)
        u_bc_t = torch.tensor(u_bc, dtype=torch.float32, device=device)

        res_f = pde_residual(model, X_f_t, D)
        u_ic_pred = model(X_ic_t)
        u_bc_pred = model(X_bc_t)

        mse_f = torch.mean(res_f ** 2)
        mse_ic = torch.mean((u_ic_pred - u_ic_t) ** 2)
        mse_bc = torch.mean((u_bc_pred - u_bc_t) ** 2)
        base_loss = mse_f + mse_ic + mse_bc
        adaptive_loss = adaptive.adaptive_term(mse_f, mse_ic, mse_bc)
        loss_model = base_loss + beta_adaptive * adaptive_loss
        optimizer_model.zero_grad()
        loss_model.backward()
        optimizer_model.step()


        res_f_w = pde_residual(model, X_f_t, D)
        u_ic_pred_w = model(X_ic_t)
        u_bc_pred_w = model(X_bc_t)
        mse_f_w = torch.mean(res_f_w ** 2)
        mse_ic_w = torch.mean((u_ic_pred_w - u_ic_t) ** 2)
        mse_bc_w = torch.mean((u_bc_pred_w - u_bc_t) ** 2)
        loss_lambda = adaptive.adaptive_term(mse_f_w, mse_ic_w, mse_bc_w)
        optimizer_lambda.zero_grad()
        (-loss_lambda).backward()
        optimizer_lambda.step()

        with torch.no_grad():
            adaptive.raw.clamp_(min=-8.0, max=8.0)

        lam = adaptive.lambdas()
        lambda_history.append(lam.copy())
        loss_history.append(loss_model.item())
        mse_f_history.append(mse_f.item())
        mse_ic_history.append(mse_ic.item())
        mse_bc_history.append(mse_bc.item())

        if epoch % 500 == 0 or epoch == 1:
            print(
                f"stage1 epoch {epoch}/{stage1_epochs} "
                f"loss={loss_model.item():.6e} mse_f={mse_f.item():.6e} "
                f"mse_ic={mse_ic.item():.6e} mse_bc={mse_bc.item():.6e} "
                f"lambda_f={lam[0]:.3e} lambda_ic={lam[1]:.3e} lambda_bc={lam[2]:.3e}"
            )


    lambda_f, lambda_ic, lambda_bc = adaptive.lambdas().tolist()
    print(
        "fixed lambdas for L-BFGS:",
        f"lambda_f={lambda_f:.6e}, lambda_ic={lambda_ic:.6e}, lambda_bc={lambda_bc:.6e}",
    )

    X_f_lb, X_ic_lb, u_ic_lb, X_bc_lb, u_bc_lb = sampler(
        n_f=n_f, n_ic=n_ic, n_bc=n_bc, T=T, x_lb=x_lb, x_rb=x_rb
    )
    X_f_lb_t = torch.tensor(X_f_lb, dtype=torch.float32, device=device)
    X_ic_lb_t = torch.tensor(X_ic_lb, dtype=torch.float32, device=device)
    u_ic_lb_t = torch.tensor(u_ic_lb, dtype=torch.float32, device=device)
    X_bc_lb_t = torch.tensor(X_bc_lb, dtype=torch.float32, device=device)
    u_bc_lb_t = torch.tensor(u_bc_lb, dtype=torch.float32, device=device)

    if stage2_lbfgs_iters > 0:
        opt_lbfgs = optim.LBFGS(
            model.parameters(),
            max_iter=stage2_lbfgs_iters,
            tolerance_grad=1e-9,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        def closure_lbfgs():
            opt_lbfgs.zero_grad()
            res_f = pde_residual(model, X_f_lb_t, D)
            u_ic_pred = model(X_ic_lb_t)
            u_bc_pred = model(X_bc_lb_t)

            mse_f = torch.mean(res_f ** 2)
            mse_ic = torch.mean((u_ic_pred - u_ic_lb_t) ** 2)
            mse_bc = torch.mean((u_bc_pred - u_bc_lb_t) ** 2)
            base_loss = mse_f + mse_ic + mse_bc
            adaptive_loss = lambda_f * mse_f + lambda_ic * mse_ic + lambda_bc * mse_bc
            loss_val = base_loss + beta_adaptive * adaptive_loss
            loss_val.backward()
            return loss_val

        print("starting stage2 L-BFGS")
        opt_lbfgs.step(closure_lbfgs)

    nt = 200
    nx = 201
    t_grid = np.linspace(0.0, T, nt)
    x_grid = np.linspace(x_lb, x_rb, nx)
    TT, XX = np.meshgrid(t_grid, x_grid)
    tx = np.stack([TT.ravel(), XX.ravel()], axis=1)
    tx_t = torch.tensor(tx, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        U_pred = model(tx_t).cpu().numpy().reshape(nx, nt)

    U_ref = solve_reference(D, T, nx, nt, x_lb=x_lb, x_rb=x_rb)
    rel_l2 = np.linalg.norm(U_pred - U_ref) / (np.linalg.norm(U_ref) + 1e-12)
    print(f"relative L2 error={rel_l2:.6e}")


    np.save(os.path.join(out_dir, "lambda_history.npy"), np.array(lambda_history))
    np.save(os.path.join(out_dir, "loss_history.npy"), np.array(loss_history))
    np.save(os.path.join(out_dir, "mse_f_history.npy"), np.array(mse_f_history))
    np.save(os.path.join(out_dir, "mse_ic_history.npy"), np.array(mse_ic_history))
    np.save(os.path.join(out_dir, "mse_bc_history.npy"), np.array(mse_bc_history))

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.title("Reference (t-x)")
    pcm1 = plt.pcolormesh(TT, XX, U_ref, shading="auto", cmap="RdYlBu")
    plt.colorbar(pcm1)
    plt.xlabel("t")
    plt.ylabel("x")

    plt.subplot(2, 2, 2)
    plt.title("Self-adaptive PINN prediction (t-x)")
    pcm2 = plt.pcolormesh(TT, XX, U_pred, shading="auto", cmap="RdYlBu")
    plt.colorbar(pcm2)
    plt.xlabel("t")
    plt.ylabel("x")

    plt.subplot(2, 2, 3)
    plt.title("Difference (PINN - ref)")
    pcm3 = plt.pcolormesh(TT, XX, U_pred - U_ref, shading="auto", cmap="bwr")
    plt.colorbar(pcm3)
    plt.xlabel("t")
    plt.ylabel("x")
    plt.tight_layout()

    fig_path = os.path.join(out_dir, "alan_cahn_self_adaptive_pinn_compare.png")
    plt.savefig(fig_path)
    print("saved comparison plot to", fig_path)

    plt.figure(figsize=(7, 4))
    lam_arr = np.array(lambda_history)
    plt.plot(lam_arr[:, 0], label="lambda_f")
    plt.plot(lam_arr[:, 1], label="lambda_ic")
    plt.plot(lam_arr[:, 2], label="lambda_bc")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("adaptive weights")
    plt.title("Adaptive weights history")
    plt.legend()
    plt.tight_layout()
    lam_fig = os.path.join(out_dir, "alan_cahn_self_adaptive_lambdas.png")
    plt.savefig(lam_fig)
    print("saved lambda plot to", lam_fig)
    return rel_l2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Self-adaptive PINN for Allen-Cahn equation (no hard constraints / no hPINN)"
    )
    parser.add_argument("--D", type=float, default=1e-4)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--x_lb", type=float, default=-1.0)
    parser.add_argument("--x_rb", type=float, default=1.0)
    parser.add_argument("--n_f", type=int, default=20000)
    parser.add_argument("--n_ic", type=int, default=300)
    parser.add_argument("--n_bc", type=int, default=300)
    parser.add_argument("--stage1_epochs", type=int, default=12000)
    parser.add_argument("--stage2_lbfgs_iters", type=int, default=1200)
    parser.add_argument("--lr_model", type=float, default=1e-3)
    parser.add_argument("--lr_lambda", type=float, default=2e-3)
    parser.add_argument("--beta_adaptive", type=float, default=1.0)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    device = set_device()
    print("device:", device)
    train_alancahn_self_adaptive_pinn(
        device=device,
        out_dir=args.out_dir,
        D=args.D,
        T=args.T,
        x_lb=args.x_lb,
        x_rb=args.x_rb,
        n_f=args.n_f,
        n_ic=args.n_ic,
        n_bc=args.n_bc,
        stage1_epochs=args.stage1_epochs,
        stage2_lbfgs_iters=args.stage2_lbfgs_iters,
        lr_model=args.lr_model,
        lr_lambda=args.lr_lambda,
        beta_adaptive=args.beta_adaptive,
    )


