import argparse
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


class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=1, hidden=96, n_layers=4):
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


def exact_solution_torch(t, x, beta):
    return (t ** beta) * torch.sin(np.pi * x)


def exact_solution_np(t, x, beta):
    return (t ** beta) * np.sin(np.pi * x)


def source_torch(t, x, alpha, beta):

    c = math.gamma(beta + 1.0) / math.gamma(beta + 1.0 - alpha)
    return (c * (t ** (beta - alpha)) + (np.pi ** 2) * (t ** beta)) * torch.sin(np.pi * x)


def caputo_l1_from_history(u_hist, alpha, dt):

    n = u_hist.shape[1] - 1
    if n <= 0:
        return torch.zeros((u_hist.shape[0], 1), dtype=u_hist.dtype, device=u_hist.device)

    diffs = u_hist[:, 1:] - u_hist[:, :-1]
    k = torch.arange(0, n, device=u_hist.device, dtype=u_hist.dtype)
    b = (k + 1.0) ** (1.0 - alpha) - k ** (1.0 - alpha)

    weights = torch.flip(b, dims=[0]).view(1, n)

    coef = (dt ** (-alpha)) / math.gamma(2.0 - alpha)
    cap = coef * torch.sum(diffs * weights, dim=1, keepdim=True)
    return cap


def build_training_points(
    x_lb,
    x_rb,
    t_lb,
    t_rb,
    n_time_steps,
    n_f_per_t,
    n_ic,
    n_bc,
    device,
):
    dt = (t_rb - t_lb) / n_time_steps
    t_grid = np.linspace(t_lb, t_rb, n_time_steps + 1, dtype=np.float32)
    t_grid_t = torch.tensor(t_grid, dtype=torch.float32, device=device)


    x_f_list = []
    for _ in range(1, n_time_steps + 1):
        x_np = np.random.uniform(x_lb, x_rb, size=(n_f_per_t, 1)).astype(np.float32)
        x_f_list.append(torch.tensor(x_np, dtype=torch.float32, device=device))


    x_ic_np = np.random.uniform(x_lb, x_rb, size=(n_ic, 1)).astype(np.float32)
    t_ic_np = np.zeros((n_ic, 1), dtype=np.float32)
    tx_ic_np = np.hstack([t_ic_np, x_ic_np]).astype(np.float32)
    tx_ic = torch.tensor(tx_ic_np, dtype=torch.float32, device=device)
    u_ic = torch.zeros((n_ic, 1), dtype=torch.float32, device=device)


    t_bc_np = np.random.uniform(t_lb, t_rb, size=(n_bc, 1)).astype(np.float32)
    tx_bc_l = torch.tensor(
        np.hstack([t_bc_np, np.ones((n_bc, 1), dtype=np.float32) * x_lb]).astype(np.float32),
        dtype=torch.float32,
        device=device,
    )
    tx_bc_r = torch.tensor(
        np.hstack([t_bc_np, np.ones((n_bc, 1), dtype=np.float32) * x_rb]).astype(np.float32),
        dtype=torch.float32,
        device=device,
    )
    u_bc = torch.zeros((n_bc, 1), dtype=torch.float32, device=device)
    return dt, t_grid_t, x_f_list, tx_ic, u_ic, tx_bc_l, tx_bc_r, u_bc


def fractional_pde_loss(model, alpha, beta, dt, t_grid_t, x_f_list):
    n_time_steps = len(x_f_list)
    mse_terms = []

    for n in range(1, n_time_steps + 1):
        x_n = x_f_list[n - 1]
        bsz = x_n.shape[0]


        t_hist = t_grid_t[: n + 1].view(1, n + 1).repeat(bsz, 1)
        x_hist = x_n.repeat(1, n + 1)
        tx_hist = torch.stack([t_hist, x_hist], dim=2).reshape(-1, 2)
        u_hist = model(tx_hist).reshape(bsz, n + 1)
        cap = caputo_l1_from_history(u_hist, alpha=alpha, dt=dt)


        t_n = t_grid_t[n]
        tx_n = torch.cat(
            [torch.full((bsz, 1), t_n.item(), dtype=torch.float32, device=x_n.device), x_n],
            dim=1,
        ).clone().detach().requires_grad_(True)
        u_n = model(tx_n)
        grads = torch.autograd.grad(
            u_n, tx_n, grad_outputs=torch.ones_like(u_n), create_graph=True
        )[0]
        u_x = grads[:, 1:2]
        u_xx = torch.autograd.grad(
            u_x, tx_n, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0][:, 1:2]

        t_col = tx_n[:, 0:1]
        x_col = tx_n[:, 1:2]
        f_n = source_torch(t_col, x_col, alpha=alpha, beta=beta)
        r = cap - u_xx - f_n
        mse_terms.append(torch.mean(r ** 2))

    return torch.mean(torch.stack(mse_terms))


def train_fpinn_time_fractional_diffusion_1d(
    alpha=0.6,
    beta=2.0,
    x_lb=0.0,
    x_rb=1.0,
    t_lb=0.0,
    t_rb=1.0,
    n_time_steps=25,
    n_f_per_t=64,
    n_ic=256,
    n_bc=256,
    adam_epochs=3000,
    lbfgs_iters=300,
    lr=1e-3,
    lambda_ic=10.0,
    lambda_bc=10.0,
    hidden=96,
    n_layers=4,
    out_dir=None,
    device=None,
):
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    if beta <= alpha:
        raise ValueError("beta must be > alpha")

    if device is None:
        device = set_device()
    if out_dir is None:
        out_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "ode_solvers_outputs", "pinns")
        )
    os.makedirs(out_dir, exist_ok=True)

    model = MLP(in_dim=2, out_dim=1, hidden=hidden, n_layers=n_layers).to(device)
    dt, t_grid_t, x_f_list, tx_ic, u_ic, tx_bc_l, tx_bc_r, u_bc = build_training_points(
        x_lb=x_lb,
        x_rb=x_rb,
        t_lb=t_lb,
        t_rb=t_rb,
        n_time_steps=n_time_steps,
        n_f_per_t=n_f_per_t,
        n_ic=n_ic,
        n_bc=n_bc,
        device=device,
    )

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    mse_pde_history = []
    mse_ic_history = []
    mse_bc_history = []

    for epoch in range(1, adam_epochs + 1):
        model.train()
        mse_pde = fractional_pde_loss(
            model=model,
            alpha=alpha,
            beta=beta,
            dt=dt,
            t_grid_t=t_grid_t,
            x_f_list=x_f_list,
        )
        u_ic_pred = model(tx_ic)
        mse_ic = torch.mean((u_ic_pred - u_ic) ** 2)

        u_bc_l = model(tx_bc_l)
        u_bc_r = model(tx_bc_r)
        mse_bc = 0.5 * (
            torch.mean((u_bc_l - u_bc) ** 2) + torch.mean((u_bc_r - u_bc) ** 2)
        )
        loss = mse_pde + lambda_ic * mse_ic + lambda_bc * mse_bc

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_history.append(loss.item())
        mse_pde_history.append(mse_pde.item())
        mse_ic_history.append(mse_ic.item())
        mse_bc_history.append(mse_bc.item())

        if epoch == 1 or epoch % 300 == 0:
            print(
                f"adam epoch {epoch}/{adam_epochs}, loss={loss.item():.6e}, "
                f"mse_pde={mse_pde.item():.6e}, mse_ic={mse_ic.item():.6e}, mse_bc={mse_bc.item():.6e}"
            )

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
            mse_pde = fractional_pde_loss(
                model=model,
                alpha=alpha,
                beta=beta,
                dt=dt,
                t_grid_t=t_grid_t,
                x_f_list=x_f_list,
            )
            u_ic_pred = model(tx_ic)
            mse_ic = torch.mean((u_ic_pred - u_ic) ** 2)
            u_bc_l = model(tx_bc_l)
            u_bc_r = model(tx_bc_r)
            mse_bc = 0.5 * (
                torch.mean((u_bc_l - u_bc) ** 2) + torch.mean((u_bc_r - u_bc) ** 2)
            )
            loss_val = mse_pde + lambda_ic * mse_ic + lambda_bc * mse_bc
            loss_val.backward()
            return loss_val

        print("starting L-BFGS stage")
        opt_lbfgs.step(closure)


    nt = 101
    nx = 201
    t_grid = np.linspace(t_lb, t_rb, nt, dtype=np.float32)
    x_grid = np.linspace(x_lb, x_rb, nx, dtype=np.float32)
    TT, XX = np.meshgrid(t_grid, x_grid)
    tx_eval = np.stack([TT.ravel(), XX.ravel()], axis=1).astype(np.float32)
    tx_eval_t = torch.tensor(tx_eval, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        U_pred = model(tx_eval_t).cpu().numpy().reshape(nx, nt)
    U_ref = exact_solution_np(TT, XX, beta=beta)
    rel_l2 = np.linalg.norm(U_pred - U_ref) / (np.linalg.norm(U_ref) + 1e-12)
    max_abs = np.max(np.abs(U_pred - U_ref))
    print(f"relative L2 error={rel_l2:.6e}")
    print(f"max absolute error={max_abs:.6e}")

    np.save(os.path.join(out_dir, "fpinn_tfde_loss.npy"), np.array(loss_history))
    np.save(os.path.join(out_dir, "fpinn_tfde_mse_pde.npy"), np.array(mse_pde_history))
    np.save(os.path.join(out_dir, "fpinn_tfde_mse_ic.npy"), np.array(mse_ic_history))
    np.save(os.path.join(out_dir, "fpinn_tfde_mse_bc.npy"), np.array(mse_bc_history))
    np.savez(
        os.path.join(out_dir, "fpinn_tfde_pred.npz"),
        TT=TT,
        XX=XX,
        U_pred=U_pred,
        U_ref=U_ref,
        U_err=U_pred - U_ref,
        alpha=np.array([alpha]),
        beta=np.array([beta]),
    )
    torch.save(model.state_dict(), os.path.join(out_dir, "fpinn_tfde_1d.pt"))

    plt.figure(figsize=(13, 8))
    plt.subplot(2, 2, 1)
    plt.title("Reference u(t,x)")
    pcm1 = plt.pcolormesh(TT, XX, U_ref, shading="auto", cmap="RdYlBu")
    plt.colorbar(pcm1)
    plt.xlabel("t")
    plt.ylabel("x")

    plt.subplot(2, 2, 2)
    plt.title("fPINN prediction u(t,x)")
    pcm2 = plt.pcolormesh(TT, XX, U_pred, shading="auto", cmap="RdYlBu")
    plt.colorbar(pcm2)
    plt.xlabel("t")
    plt.ylabel("x")

    plt.subplot(2, 2, 3)
    plt.title("Error (fPINN - ref)")
    pcm3 = plt.pcolormesh(TT, XX, U_pred - U_ref, shading="auto", cmap="bwr")
    plt.colorbar(pcm3)
    plt.xlabel("t")
    plt.ylabel("x")

    plt.subplot(2, 2, 4)
    idx = np.linspace(0, nt - 1, 5, dtype=int)
    for k in idx:
        plt.plot(x_grid, U_ref[:, k], "-", lw=1.8, label=f"ref t={t_grid[k]:.2f}")
        plt.plot(x_grid, U_pred[:, k], "--", lw=1.4, label=f"fPINN t={t_grid[k]:.2f}")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("Slices at several times")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "fpinn_tfde_1d_compare.png")
    plt.savefig(fig_path, dpi=180)
    print("saved figure to", fig_path)

    plt.figure(figsize=(8, 4.8))
    plt.plot(loss_history, label="total", lw=1.5)
    plt.plot(mse_pde_history, label="mse_pde", lw=1.2)
    plt.plot(np.array(mse_ic_history) * lambda_ic, label="lambda_ic*mse_ic", lw=1.2)
    plt.plot(np.array(mse_bc_history) * lambda_bc, label="lambda_bc*mse_bc", lw=1.2)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss terms")
    plt.title("fPINN training history")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    loss_fig = os.path.join(out_dir, "fpinn_tfde_1d_loss.png")
    plt.savefig(loss_fig, dpi=180)
    print("saved loss figure to", loss_fig)
    return rel_l2, max_abs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="fPINN for 1D time-fractional diffusion equation (Caputo in time)"
    )
    parser.add_argument("--alpha", type=float, default=0.6, help="fractional order in (0,1)")
    parser.add_argument("--beta", type=float, default=2.0, help="exact-solution time exponent (>alpha)")
    parser.add_argument("--x_lb", type=float, default=0.0)
    parser.add_argument("--x_rb", type=float, default=1.0)
    parser.add_argument("--t_lb", type=float, default=0.0)
    parser.add_argument("--t_rb", type=float, default=1.0)
    parser.add_argument("--n_time_steps", type=int, default=25, help="time grid count for L1 Caputo")
    parser.add_argument("--n_f_per_t", type=int, default=64, help="PDE collocation samples per time level")
    parser.add_argument("--n_ic", type=int, default=256)
    parser.add_argument("--n_bc", type=int, default=256)
    parser.add_argument("--adam_epochs", type=int, default=3000)
    parser.add_argument("--lbfgs_iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_ic", type=float, default=10.0)
    parser.add_argument("--lambda_bc", type=float, default=10.0)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    device = set_device()
    print("device:", device)
    train_fpinn_time_fractional_diffusion_1d(
        alpha=args.alpha,
        beta=args.beta,
        x_lb=args.x_lb,
        x_rb=args.x_rb,
        t_lb=args.t_lb,
        t_rb=args.t_rb,
        n_time_steps=args.n_time_steps,
        n_f_per_t=args.n_f_per_t,
        n_ic=args.n_ic,
        n_bc=args.n_bc,
        adam_epochs=args.adam_epochs,
        lbfgs_iters=args.lbfgs_iters,
        lr=args.lr,
        lambda_ic=args.lambda_ic,
        lambda_bc=args.lambda_bc,
        hidden=args.hidden,
        n_layers=args.n_layers,
        out_dir=args.out_dir,
        device=device,
    )


