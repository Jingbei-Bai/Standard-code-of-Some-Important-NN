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


class CPINN1DAdvection(nn.Module):


    def __init__(self, x_lb=-1.0, x_rb=1.0, t_lb=0.0, t_rb=1.0, n_subdomains=2, hidden=96, n_layers=4):
        super().__init__()
        if n_subdomains < 2:
            raise ValueError("cPINN needs at least 2 subdomains")
        edges = np.linspace(x_lb, x_rb, n_subdomains + 1, dtype=np.float32)
        self.register_buffer("edges", torch.tensor(edges, dtype=torch.float32))
        self.register_buffer("t_lb", torch.tensor([t_lb], dtype=torch.float32))
        self.register_buffer("t_rb", torch.tensor([t_rb], dtype=torch.float32))
        self.subnets = nn.ModuleList(
            [MLP(in_dim=2, out_dim=1, hidden=hidden, n_layers=n_layers) for _ in range(n_subdomains)]
        )

    @property
    def n_subdomains(self):
        return len(self.subnets)

    def sub_u(self, sub_idx, tx):
        t = tx[:, 0:1]
        x = tx[:, 1:2]
        left = self.edges[sub_idx]
        right = self.edges[sub_idx + 1]
        xc = 0.5 * (left + right)
        hx = 0.5 * (right - left)
        tmid = 0.5 * (self.t_lb + self.t_rb)
        ht = 0.5 * (self.t_rb - self.t_lb)


        x_loc = (x - xc) / hx
        t_loc = (t - tmid) / ht
        tx_loc = torch.cat([t_loc, x_loc], dim=1)
        return self.subnets[sub_idx](tx_loc)


def initial_condition_torch(x):
    return torch.sin(np.pi * x)


def exact_solution_torch(t, x, c):
    return torch.sin(np.pi * (x - c * t))


def exact_solution_np(t, x, c):
    return np.sin(np.pi * (x - c * t))


def sample_subdomain_tx(edges, t_lb, t_rb, n_per_sub):
    tx_list = []
    for i in range(len(edges) - 1):
        left = edges[i]
        right = edges[i + 1]
        t = np.random.uniform(t_lb, t_rb, size=(n_per_sub, 1)).astype(np.float32)
        x = np.random.uniform(left, right, size=(n_per_sub, 1)).astype(np.float32)
        tx_list.append(np.hstack([t, x]).astype(np.float32))
    return tx_list


def sample_subdomain_ic(edges, n_per_sub):
    tx_list = []
    for i in range(len(edges) - 1):
        left = edges[i]
        right = edges[i + 1]
        t = np.zeros((n_per_sub, 1), dtype=np.float32)
        x = np.random.uniform(left, right, size=(n_per_sub, 1)).astype(np.float32)
        tx_list.append(np.hstack([t, x]).astype(np.float32))
    return tx_list


def sample_interface_tx(edges, t_lb, t_rb, n_if):
    tx_if_list = []
    for j in range(1, len(edges) - 1):
        t = np.random.uniform(t_lb, t_rb, size=(n_if, 1)).astype(np.float32)
        x = np.ones((n_if, 1), dtype=np.float32) * edges[j]
        tx_if_list.append(np.hstack([t, x]).astype(np.float32))
    return tx_if_list


def pde_residual(model, sub_idx, tx, c):
    tx = tx.clone().detach().requires_grad_(True)
    u = model.sub_u(sub_idx, tx)
    grads = torch.autograd.grad(
        u, tx, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_t = grads[:, 0:1]
    u_x = grads[:, 1:2]
    return u_t + c * u_x


def compute_losses(
    model,
    c,
    tx_f_list,
    tx_ic_list,
    tx_if_list,
    tx_bc_in,
    x_inflow,
):
    mse_pde_terms = []
    mse_ic_terms = []
    for i in range(model.n_subdomains):
        r = pde_residual(model, i, tx_f_list[i], c=c)
        mse_pde_terms.append(torch.mean(r ** 2))

        u_ic_pred = model.sub_u(i, tx_ic_list[i])
        u_ic_true = initial_condition_torch(tx_ic_list[i][:, 1:2])
        mse_ic_terms.append(torch.mean((u_ic_pred - u_ic_true) ** 2))

    mse_pde = torch.mean(torch.stack(mse_pde_terms))
    mse_ic = torch.mean(torch.stack(mse_ic_terms))

    if c >= 0.0:
        inflow_sub = 0
    else:
        inflow_sub = model.n_subdomains - 1
    u_bc_pred = model.sub_u(inflow_sub, tx_bc_in)
    t_bc = tx_bc_in[:, 0:1]
    x_bc = torch.ones_like(t_bc) * x_inflow
    u_bc_true = exact_solution_torch(t_bc, x_bc, c=c)
    mse_bc = torch.mean((u_bc_pred - u_bc_true) ** 2)

    mse_if_u_terms = []
    mse_if_flux_terms = []
    for j in range(1, model.n_subdomains):
        tx_if = tx_if_list[j - 1]
        u_l = model.sub_u(j - 1, tx_if)
        u_r = model.sub_u(j, tx_if)
        flux_l = c * u_l
        flux_r = c * u_r
        mse_if_u_terms.append(torch.mean((u_l - u_r) ** 2))
        mse_if_flux_terms.append(torch.mean((flux_l - flux_r) ** 2))

    mse_if_u = torch.mean(torch.stack(mse_if_u_terms))
    mse_if_flux = torch.mean(torch.stack(mse_if_flux_terms))
    return mse_pde, mse_ic, mse_bc, mse_if_u, mse_if_flux


def predict_piecewise(model, tx_np, device):
    x = tx_np[:, 1]
    edges = model.edges.detach().cpu().numpy()
    u_pred = np.zeros(tx_np.shape[0], dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for i in range(model.n_subdomains):
            left = edges[i]
            right = edges[i + 1]
            if i < model.n_subdomains - 1:
                mask = (x >= left) & (x < right)
            else:
                mask = (x >= left) & (x <= right)
            if np.any(mask):
                tx_i = torch.tensor(tx_np[mask], dtype=torch.float32, device=device)
                u_i = model.sub_u(i, tx_i).cpu().numpy().squeeze()
                u_pred[mask] = u_i
    return u_pred


def interface_jump_metrics(model, c, t_lb, t_rb, n_eval_if=200):
    device = model.edges.device
    ts = np.linspace(t_lb, t_rb, n_eval_if, dtype=np.float32)[:, None]
    jump_u_all = []
    jump_flux_all = []
    for j in range(1, model.n_subdomains):
        x_if = model.edges[j].item()
        x_col = np.ones_like(ts) * x_if
        tx = torch.tensor(np.hstack([ts, x_col]), dtype=torch.float32, device=device)
        with torch.no_grad():
            u_l = model.sub_u(j - 1, tx)
            u_r = model.sub_u(j, tx)
            jump_u = torch.abs(u_l - u_r).cpu().numpy().squeeze()
            jump_flux = torch.abs(c * u_l - c * u_r).cpu().numpy().squeeze()
        jump_u_all.append(jump_u)
        jump_flux_all.append(jump_flux)
    jump_u_all = np.array(jump_u_all)
    jump_flux_all = np.array(jump_flux_all)
    return float(np.max(jump_u_all)), float(np.max(jump_flux_all))


def train_cpinn_linear_advection(
    x_lb=-1.0,
    x_rb=1.0,
    t_lb=0.0,
    t_rb=1.0,
    c=1.0,
    n_subdomains=2,
    n_f_per_sub=2000,
    n_ic_per_sub=300,
    n_bc=300,
    n_if=100,
    adam_epochs=5000,
    lbfgs_iters=400,
    lr=1e-3,
    lambda_ic=10.0,
    lambda_bc=10.0,
    lambda_if_u=5.0,
    lambda_if_flux=20.0,
    hidden=96,
    n_layers=4,
    out_dir=None,
    device=None,
):
    if device is None:
        device = set_device()
    if out_dir is None:
        out_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "ode_solvers_outputs", "pinns")
        )
    os.makedirs(out_dir, exist_ok=True)

    model = CPINN1DAdvection(
        x_lb=x_lb,
        x_rb=x_rb,
        t_lb=t_lb,
        t_rb=t_rb,
        n_subdomains=n_subdomains,
        hidden=hidden,
        n_layers=n_layers,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    edges_np = model.edges.detach().cpu().numpy()

    if c >= 0.0:
        x_inflow = x_lb
    else:
        x_inflow = x_rb

    loss_history = []
    mse_pde_history = []
    mse_ic_history = []
    mse_bc_history = []
    mse_if_u_history = []
    mse_if_flux_history = []

    for epoch in range(1, adam_epochs + 1):
        model.train()
        tx_f_list_np = sample_subdomain_tx(edges_np, t_lb=t_lb, t_rb=t_rb, n_per_sub=n_f_per_sub)
        tx_ic_list_np = sample_subdomain_ic(edges_np, n_per_sub=n_ic_per_sub)
        tx_if_list_np = sample_interface_tx(edges_np, t_lb=t_lb, t_rb=t_rb, n_if=n_if)
        t_bc = np.random.uniform(t_lb, t_rb, size=(n_bc, 1)).astype(np.float32)
        x_bc = np.ones((n_bc, 1), dtype=np.float32) * x_inflow
        tx_bc_np = np.hstack([t_bc, x_bc]).astype(np.float32)

        tx_f_list = [torch.tensor(a, dtype=torch.float32, device=device) for a in tx_f_list_np]
        tx_ic_list = [torch.tensor(a, dtype=torch.float32, device=device) for a in tx_ic_list_np]
        tx_if_list = [torch.tensor(a, dtype=torch.float32, device=device) for a in tx_if_list_np]
        tx_bc_in = torch.tensor(tx_bc_np, dtype=torch.float32, device=device)

        mse_pde, mse_ic, mse_bc, mse_if_u, mse_if_flux = compute_losses(
            model=model,
            c=c,
            tx_f_list=tx_f_list,
            tx_ic_list=tx_ic_list,
            tx_if_list=tx_if_list,
            tx_bc_in=tx_bc_in,
            x_inflow=x_inflow,
        )
        loss = (
            mse_pde
            + lambda_ic * mse_ic
            + lambda_bc * mse_bc
            + lambda_if_u * mse_if_u
            + lambda_if_flux * mse_if_flux
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_history.append(loss.item())
        mse_pde_history.append(mse_pde.item())
        mse_ic_history.append(mse_ic.item())
        mse_bc_history.append(mse_bc.item())
        mse_if_u_history.append(mse_if_u.item())
        mse_if_flux_history.append(mse_if_flux.item())

        if epoch == 1 or epoch % 500 == 0:
            print(
                f"adam epoch {epoch}/{adam_epochs}, loss={loss.item():.6e}, "
                f"mse_pde={mse_pde.item():.6e}, mse_ic={mse_ic.item():.6e}, "
                f"mse_bc={mse_bc.item():.6e}, mse_if_u={mse_if_u.item():.6e}, "
                f"mse_if_flux={mse_if_flux.item():.6e}"
            )

    if lbfgs_iters > 0:
        tx_f_list_np = sample_subdomain_tx(edges_np, t_lb=t_lb, t_rb=t_rb, n_per_sub=n_f_per_sub)
        tx_ic_list_np = sample_subdomain_ic(edges_np, n_per_sub=n_ic_per_sub)
        tx_if_list_np = sample_interface_tx(edges_np, t_lb=t_lb, t_rb=t_rb, n_if=n_if)
        t_bc = np.random.uniform(t_lb, t_rb, size=(n_bc, 1)).astype(np.float32)
        x_bc = np.ones((n_bc, 1), dtype=np.float32) * x_inflow
        tx_bc_np = np.hstack([t_bc, x_bc]).astype(np.float32)

        tx_f_list = [torch.tensor(a, dtype=torch.float32, device=device) for a in tx_f_list_np]
        tx_ic_list = [torch.tensor(a, dtype=torch.float32, device=device) for a in tx_ic_list_np]
        tx_if_list = [torch.tensor(a, dtype=torch.float32, device=device) for a in tx_if_list_np]
        tx_bc_in = torch.tensor(tx_bc_np, dtype=torch.float32, device=device)

        opt_lbfgs = optim.LBFGS(
            model.parameters(),
            max_iter=lbfgs_iters,
            tolerance_grad=1e-10,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt_lbfgs.zero_grad()
            mse_pde, mse_ic, mse_bc, mse_if_u, mse_if_flux = compute_losses(
                model=model,
                c=c,
                tx_f_list=tx_f_list,
                tx_ic_list=tx_ic_list,
                tx_if_list=tx_if_list,
                tx_bc_in=tx_bc_in,
                x_inflow=x_inflow,
            )
            loss_val = (
                mse_pde
                + lambda_ic * mse_ic
                + lambda_bc * mse_bc
                + lambda_if_u * mse_if_u
                + lambda_if_flux * mse_if_flux
            )
            loss_val.backward()
            return loss_val

        print("starting L-BFGS stage")
        opt_lbfgs.step(closure)

    nt = 201
    nx = 301
    t_grid = np.linspace(t_lb, t_rb, nt, dtype=np.float32)
    x_grid = np.linspace(x_lb, x_rb, nx, dtype=np.float32)
    TT, XX = np.meshgrid(t_grid, x_grid)
    tx_eval = np.stack([TT.ravel(), XX.ravel()], axis=1).astype(np.float32)
    u_pred = predict_piecewise(model, tx_eval, device=device).reshape(nx, nt)
    u_ref = exact_solution_np(TT, XX, c=c)

    rel_l2 = np.linalg.norm(u_pred - u_ref) / (np.linalg.norm(u_ref) + 1e-12)
    max_abs = np.max(np.abs(u_pred - u_ref))
    jump_u_max, jump_flux_max = interface_jump_metrics(
        model, c=c, t_lb=t_lb, t_rb=t_rb, n_eval_if=400
    )
    print(f"relative L2 error={rel_l2:.6e}")
    print(f"max absolute error={max_abs:.6e}")
    print(f"interface jump (u) max={jump_u_max:.6e}")
    print(f"interface jump (flux) max={jump_flux_max:.6e}")

    np.save(os.path.join(out_dir, "cpinn_advection_loss.npy"), np.array(loss_history))
    np.save(os.path.join(out_dir, "cpinn_advection_mse_pde.npy"), np.array(mse_pde_history))
    np.save(os.path.join(out_dir, "cpinn_advection_mse_ic.npy"), np.array(mse_ic_history))
    np.save(os.path.join(out_dir, "cpinn_advection_mse_bc.npy"), np.array(mse_bc_history))
    np.save(os.path.join(out_dir, "cpinn_advection_mse_if_u.npy"), np.array(mse_if_u_history))
    np.save(os.path.join(out_dir, "cpinn_advection_mse_if_flux.npy"), np.array(mse_if_flux_history))
    np.savez(
        os.path.join(out_dir, "cpinn_advection_pred.npz"),
        TT=TT,
        XX=XX,
        U_pred=u_pred,
        U_ref=u_ref,
        U_err=u_pred - u_ref,
    )
    np.save(
        os.path.join(out_dir, "cpinn_advection_interface_jumps.npy"),
        np.array([jump_u_max, jump_flux_max], dtype=np.float64),
    )
    torch.save(model.state_dict(), os.path.join(out_dir, "cpinn_linear_advection_1d.pt"))

    plt.figure(figsize=(13, 8))
    plt.subplot(2, 2, 1)
    plt.title("Reference u(t,x)")
    pcm1 = plt.pcolormesh(TT, XX, u_ref, shading="auto", cmap="RdYlBu")
    plt.colorbar(pcm1)
    plt.xlabel("t")
    plt.ylabel("x")

    plt.subplot(2, 2, 2)
    plt.title("cPINN prediction u(t,x)")
    pcm2 = plt.pcolormesh(TT, XX, u_pred, shading="auto", cmap="RdYlBu")
    plt.colorbar(pcm2)
    plt.xlabel("t")
    plt.ylabel("x")

    plt.subplot(2, 2, 3)
    plt.title("Error (cPINN - ref)")
    pcm3 = plt.pcolormesh(TT, XX, u_pred - u_ref, shading="auto", cmap="bwr")
    plt.colorbar(pcm3)
    plt.xlabel("t")
    plt.ylabel("x")

    plt.subplot(2, 2, 4)
    ts_idx = np.linspace(0, nt - 1, 5, dtype=int)
    for idx in ts_idx:
        plt.plot(x_grid, u_ref[:, idx], "-", lw=1.8, label=f"ref t={t_grid[idx]:.2f}")
        plt.plot(x_grid, u_pred[:, idx], "--", lw=1.5, label=f"cPINN t={t_grid[idx]:.2f}")
    for x_if in edges_np[1:-1]:
        plt.axvline(x_if, color="gray", ls="--", lw=1, alpha=0.4)
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("Slices at several times")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "cpinn_linear_advection_1d_compare.png")
    plt.savefig(fig_path, dpi=180)
    print("saved figure to", fig_path)

    plt.figure(figsize=(8, 4.8))
    plt.plot(loss_history, label="total", lw=1.5)
    plt.plot(mse_pde_history, label="mse_pde", lw=1.2)
    plt.plot(np.array(mse_ic_history) * lambda_ic, label="lambda_ic*mse_ic", lw=1.2)
    plt.plot(np.array(mse_bc_history) * lambda_bc, label="lambda_bc*mse_bc", lw=1.2)
    plt.plot(np.array(mse_if_u_history) * lambda_if_u, label="lambda_if_u*mse_if_u", lw=1.2)
    plt.plot(np.array(mse_if_flux_history) * lambda_if_flux, label="lambda_if_flux*mse_if_flux", lw=1.2)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss terms")
    plt.title("cPINN training history")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    loss_fig = os.path.join(out_dir, "cpinn_linear_advection_1d_loss.png")
    plt.savefig(loss_fig, dpi=180)
    print("saved loss figure to", loss_fig)
    return rel_l2, max_abs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cPINN for 1D linear advection equation")
    parser.add_argument("--x_lb", type=float, default=-1.0)
    parser.add_argument("--x_rb", type=float, default=1.0)
    parser.add_argument("--t_lb", type=float, default=0.0)
    parser.add_argument("--t_rb", type=float, default=1.0)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--n_subdomains", type=int, default=2)
    parser.add_argument("--n_f_per_sub", type=int, default=2000)
    parser.add_argument("--n_ic_per_sub", type=int, default=300)
    parser.add_argument("--n_bc", type=int, default=300)
    parser.add_argument("--n_if", type=int, default=100)
    parser.add_argument("--adam_epochs", type=int, default=5000)
    parser.add_argument("--lbfgs_iters", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_ic", type=float, default=10.0)
    parser.add_argument("--lambda_bc", type=float, default=10.0)
    parser.add_argument("--lambda_if_u", type=float, default=5.0)
    parser.add_argument("--lambda_if_flux", type=float, default=20.0)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    device = set_device()
    print("device:", device)
    train_cpinn_linear_advection(
        x_lb=args.x_lb,
        x_rb=args.x_rb,
        t_lb=args.t_lb,
        t_rb=args.t_rb,
        c=args.c,
        n_subdomains=args.n_subdomains,
        n_f_per_sub=args.n_f_per_sub,
        n_ic_per_sub=args.n_ic_per_sub,
        n_bc=args.n_bc,
        n_if=args.n_if,
        adam_epochs=args.adam_epochs,
        lbfgs_iters=args.lbfgs_iters,
        lr=args.lr,
        lambda_ic=args.lambda_ic,
        lambda_bc=args.lambda_bc,
        lambda_if_u=args.lambda_if_u,
        lambda_if_flux=args.lambda_if_flux,
        hidden=args.hidden,
        n_layers=args.n_layers,
        out_dir=args.out_dir,
        device=device,
    )


