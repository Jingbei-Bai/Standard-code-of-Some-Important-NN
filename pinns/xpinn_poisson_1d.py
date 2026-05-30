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
    def __init__(self, in_dim=1, out_dim=1, hidden=64, n_layers=4):
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

    def forward(self, x):
        return self.net(x)


class XPINN1D(nn.Module):


    def __init__(self, x_lb=-1.0, x_rb=1.0, n_subdomains=2, hidden=64, n_layers=4):
        super().__init__()
        if n_subdomains < 2:
            raise ValueError("xPINN requires n_subdomains >= 2")
        edges = np.linspace(x_lb, x_rb, n_subdomains + 1, dtype=np.float32)
        self.register_buffer("edges", torch.tensor(edges, dtype=torch.float32))
        self.subnets = nn.ModuleList(
            [MLP(in_dim=1, out_dim=1, hidden=hidden, n_layers=n_layers) for _ in range(n_subdomains)]
        )

    @property
    def n_subdomains(self):
        return len(self.subnets)

    def sub_u(self, sub_idx, x):
        left = self.edges[sub_idx]
        right = self.edges[sub_idx + 1]
        center = 0.5 * (left + right)
        half_len = 0.5 * (right - left)
        x_local = (x - center) / half_len
        return self.subnets[sub_idx](x_local)


def exact_solution_np(x):
    return np.sin(np.pi * x)


def forcing_torch(x):
    return (np.pi ** 2) * torch.sin(np.pi * x)


def sample_subdomain_points(edges_np, n_f_per_sub):
    xs = []
    for i in range(len(edges_np) - 1):
        left = edges_np[i]
        right = edges_np[i + 1]
        x_i = np.random.uniform(left, right, size=(n_f_per_sub, 1)).astype(np.float32)
        xs.append(x_i)
    return xs


def pde_quantities(model, sub_idx, x):
    x = x.clone().detach().requires_grad_(True)
    u = model.sub_u(sub_idx, x)
    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]
    r = -u_xx - forcing_torch(x)
    return r, u, u_x, u_xx


def compute_losses(
    model,
    x_f_list_t,
    x_lb=-1.0,
    x_rb=1.0,
    n_if=8,
):
    device = model.edges.device
    mse_pde_list = []
    for sd_idx, x_f in enumerate(x_f_list_t):
        r, _, _, _ = pde_quantities(model, sd_idx, x_f)
        mse_pde_list.append(torch.mean(r ** 2))
    mse_pde = torch.mean(torch.stack(mse_pde_list))

    x_left = torch.tensor([[x_lb]], dtype=torch.float32, device=device)
    x_right = torch.tensor([[x_rb]], dtype=torch.float32, device=device)
    u_left = model.sub_u(0, x_left)
    u_right = model.sub_u(model.n_subdomains - 1, x_right)
    mse_bc = 0.5 * ((u_left ** 2).mean() + (u_right ** 2).mean())

    mse_if_u_terms = []
    mse_if_flux_terms = []
    mse_if_r_terms = []
    for j in range(1, model.n_subdomains):
        x_if_val = model.edges[j].item()
        x_if = torch.full(
            (n_if, 1),
            fill_value=x_if_val,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        u_l = model.sub_u(j - 1, x_if)
        u_r = model.sub_u(j, x_if)
        ux_l = torch.autograd.grad(
            u_l, x_if, grad_outputs=torch.ones_like(u_l), create_graph=True
        )[0]
        ux_r = torch.autograd.grad(
            u_r, x_if, grad_outputs=torch.ones_like(u_r), create_graph=True
        )[0]
        uxx_l = torch.autograd.grad(
            ux_l, x_if, grad_outputs=torch.ones_like(ux_l), create_graph=True
        )[0]
        uxx_r = torch.autograd.grad(
            ux_r, x_if, grad_outputs=torch.ones_like(ux_r), create_graph=True
        )[0]
        r_l = -uxx_l - forcing_torch(x_if)
        r_r = -uxx_r - forcing_torch(x_if)

        mse_if_u_terms.append(torch.mean((u_l - u_r) ** 2))
        mse_if_flux_terms.append(torch.mean((ux_l - ux_r) ** 2))
        mse_if_r_terms.append(torch.mean((r_l - r_r) ** 2))

    mse_if_u = torch.mean(torch.stack(mse_if_u_terms))
    mse_if_flux = torch.mean(torch.stack(mse_if_flux_terms))
    mse_if_r = torch.mean(torch.stack(mse_if_r_terms))

    return mse_pde, mse_bc, mse_if_u, mse_if_flux, mse_if_r


def predict_piecewise(model, x_eval_np, device):
    x_flat = x_eval_np.squeeze()
    edges = model.edges.detach().cpu().numpy()
    u_pred = np.zeros_like(x_flat, dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for i in range(model.n_subdomains):
            left = edges[i]
            right = edges[i + 1]
            if i < model.n_subdomains - 1:
                mask = (x_flat >= left) & (x_flat < right)
            else:
                mask = (x_flat >= left) & (x_flat <= right)
            if np.any(mask):
                x_i_t = torch.tensor(x_flat[mask][:, None], dtype=torch.float32, device=device)
                u_i = model.sub_u(i, x_i_t).cpu().numpy().squeeze()
                u_pred[mask] = u_i
    return u_pred


def interface_jumps(model):
    device = model.edges.device
    jumps_u = []
    jumps_flux = []
    for j in range(1, model.n_subdomains):
        x_if_val = model.edges[j].item()
        x_if = torch.tensor([[x_if_val]], dtype=torch.float32, device=device, requires_grad=True)
        u_l = model.sub_u(j - 1, x_if)
        u_r = model.sub_u(j, x_if)
        ux_l = torch.autograd.grad(
            u_l, x_if, grad_outputs=torch.ones_like(u_l), create_graph=False
        )[0]
        ux_r = torch.autograd.grad(
            u_r, x_if, grad_outputs=torch.ones_like(u_r), create_graph=False
        )[0]
        jumps_u.append(float(torch.abs(u_l - u_r).item()))
        jumps_flux.append(float(torch.abs(ux_l - ux_r).item()))
    return np.array(jumps_u), np.array(jumps_flux)


def train_xpinn_poisson_1d(
    x_lb=-1.0,
    x_rb=1.0,
    n_subdomains=2,
    n_f_per_sub=1500,
    n_if=8,
    adam_epochs=5000,
    lbfgs_iters=400,
    lr=1e-3,
    lambda_bc=20.0,
    lambda_if_u=20.0,
    lambda_if_flux=20.0,
    lambda_if_r=1.0,
    hidden=64,
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

    model = XPINN1D(
        x_lb=x_lb,
        x_rb=x_rb,
        n_subdomains=n_subdomains,
        hidden=hidden,
        n_layers=n_layers,
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=lr)
    edges_np = model.edges.detach().cpu().numpy()

    loss_history = []
    mse_pde_history = []
    mse_bc_history = []
    mse_if_u_history = []
    mse_if_flux_history = []
    mse_if_r_history = []

    for epoch in range(1, adam_epochs + 1):
        model.train()
        x_f_list_np = sample_subdomain_points(edges_np, n_f_per_sub=n_f_per_sub)
        x_f_list_t = [torch.tensor(x_i, dtype=torch.float32, device=device) for x_i in x_f_list_np]

        mse_pde, mse_bc, mse_if_u, mse_if_flux, mse_if_r = compute_losses(
            model=model,
            x_f_list_t=x_f_list_t,
            x_lb=x_lb,
            x_rb=x_rb,
            n_if=n_if,
        )
        loss = (
            mse_pde
            + lambda_bc * mse_bc
            + lambda_if_u * mse_if_u
            + lambda_if_flux * mse_if_flux
            + lambda_if_r * mse_if_r
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_history.append(loss.item())
        mse_pde_history.append(mse_pde.item())
        mse_bc_history.append(mse_bc.item())
        mse_if_u_history.append(mse_if_u.item())
        mse_if_flux_history.append(mse_if_flux.item())
        mse_if_r_history.append(mse_if_r.item())

        if epoch == 1 or epoch % 500 == 0:
            print(
                f"adam epoch {epoch}/{adam_epochs}, loss={loss.item():.6e}, "
                f"mse_pde={mse_pde.item():.6e}, mse_bc={mse_bc.item():.6e}, "
                f"mse_if_u={mse_if_u.item():.6e}, mse_if_flux={mse_if_flux.item():.6e}, "
                f"mse_if_r={mse_if_r.item():.6e}"
            )

    if lbfgs_iters > 0:
        x_f_list_np = sample_subdomain_points(edges_np, n_f_per_sub=n_f_per_sub)
        x_f_list_t = [torch.tensor(x_i, dtype=torch.float32, device=device) for x_i in x_f_list_np]

        opt_lbfgs = optim.LBFGS(
            model.parameters(),
            max_iter=lbfgs_iters,
            tolerance_grad=1e-10,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt_lbfgs.zero_grad()
            mse_pde, mse_bc, mse_if_u, mse_if_flux, mse_if_r = compute_losses(
                model=model,
                x_f_list_t=x_f_list_t,
                x_lb=x_lb,
                x_rb=x_rb,
                n_if=n_if,
            )
            loss_val = (
                mse_pde
                + lambda_bc * mse_bc
                + lambda_if_u * mse_if_u
                + lambda_if_flux * mse_if_flux
                + lambda_if_r * mse_if_r
            )
            loss_val.backward()
            return loss_val

        print("starting L-BFGS stage")
        opt_lbfgs.step(closure)

    x_eval = np.linspace(x_lb, x_rb, 1001, dtype=np.float32)[:, None]
    u_pred = predict_piecewise(model, x_eval, device=device)
    u_ref = exact_solution_np(x_eval.squeeze())
    rel_l2 = np.linalg.norm(u_pred - u_ref) / (np.linalg.norm(u_ref) + 1e-12)
    max_abs = np.max(np.abs(u_pred - u_ref))
    jump_u, jump_flux = interface_jumps(model)

    print(f"relative L2 error={rel_l2:.6e}")
    print(f"max absolute error={max_abs:.6e}")
    print(f"interface |u^- - u^+| max={np.max(jump_u):.6e}")
    print(f"interface |u_x^- - u_x^+| max={np.max(jump_flux):.6e}")

    np.save(os.path.join(out_dir, "xpinn_poisson_loss.npy"), np.array(loss_history))
    np.save(os.path.join(out_dir, "xpinn_poisson_mse_pde.npy"), np.array(mse_pde_history))
    np.save(os.path.join(out_dir, "xpinn_poisson_mse_bc.npy"), np.array(mse_bc_history))
    np.save(os.path.join(out_dir, "xpinn_poisson_mse_if_u.npy"), np.array(mse_if_u_history))
    np.save(os.path.join(out_dir, "xpinn_poisson_mse_if_flux.npy"), np.array(mse_if_flux_history))
    np.save(os.path.join(out_dir, "xpinn_poisson_mse_if_r.npy"), np.array(mse_if_r_history))
    np.save(
        os.path.join(out_dir, "xpinn_poisson_pred.npy"),
        np.stack([x_eval.squeeze(), u_pred, u_ref], axis=1),
    )
    np.save(
        os.path.join(out_dir, "xpinn_poisson_interface_jumps.npy"),
        np.stack([jump_u, jump_flux], axis=1),
    )
    torch.save(model.state_dict(), os.path.join(out_dir, "xpinn_poisson_1d.pt"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    axes[0].plot(x_eval.squeeze(), u_ref, "k-", lw=2, label="exact")
    axes[0].plot(x_eval.squeeze(), u_pred, "r--", lw=2, label="xPINN")
    for x_if in edges_np[1:-1]:
        axes[0].axvline(x_if, color="gray", ls="--", lw=1, alpha=0.5)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(x)")
    axes[0].set_title("1D Poisson: xPINN solution")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(x_eval.squeeze(), np.abs(u_pred - u_ref), "b-", lw=2, label="|error|")
    for x_if in edges_np[1:-1]:
        axes[1].axvline(x_if, color="gray", ls="--", lw=1, alpha=0.5)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("absolute error")
    axes[1].set_title("Pointwise error")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "xpinn_poisson_1d_compare.png")
    plt.savefig(fig_path, dpi=180)
    print("saved figure to", fig_path)

    plt.figure(figsize=(8, 4.8))
    plt.plot(loss_history, label="total", lw=1.5)
    plt.plot(mse_pde_history, label="mse_pde", lw=1.2)
    plt.plot(np.array(mse_bc_history) * lambda_bc, label="lambda_bc*mse_bc", lw=1.2)
    plt.plot(np.array(mse_if_u_history) * lambda_if_u, label="lambda_if_u*mse_if_u", lw=1.2)
    plt.plot(np.array(mse_if_flux_history) * lambda_if_flux, label="lambda_if_flux*mse_if_flux", lw=1.2)
    plt.plot(np.array(mse_if_r_history) * lambda_if_r, label="lambda_if_r*mse_if_r", lw=1.2)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss terms")
    plt.title("xPINN training history")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    loss_fig = os.path.join(out_dir, "xpinn_poisson_1d_loss.png")
    plt.savefig(loss_fig, dpi=180)
    print("saved loss figure to", loss_fig)

    return rel_l2, max_abs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="xPINN for 1D Poisson equation")
    parser.add_argument("--x_lb", type=float, default=-1.0)
    parser.add_argument("--x_rb", type=float, default=1.0)
    parser.add_argument("--n_subdomains", type=int, default=2)
    parser.add_argument("--n_f_per_sub", type=int, default=1500)
    parser.add_argument("--n_if", type=int, default=8)
    parser.add_argument("--adam_epochs", type=int, default=5000)
    parser.add_argument("--lbfgs_iters", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_bc", type=float, default=20.0)
    parser.add_argument("--lambda_if_u", type=float, default=20.0)
    parser.add_argument("--lambda_if_flux", type=float, default=20.0)
    parser.add_argument("--lambda_if_r", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    device = set_device()
    print("device:", device)
    train_xpinn_poisson_1d(
        x_lb=args.x_lb,
        x_rb=args.x_rb,
        n_subdomains=args.n_subdomains,
        n_f_per_sub=args.n_f_per_sub,
        n_if=args.n_if,
        adam_epochs=args.adam_epochs,
        lbfgs_iters=args.lbfgs_iters,
        lr=args.lr,
        lambda_bc=args.lambda_bc,
        lambda_if_u=args.lambda_if_u,
        lambda_if_flux=args.lambda_if_flux,
        lambda_if_r=args.lambda_if_r,
        hidden=args.hidden,
        n_layers=args.n_layers,
        out_dir=args.out_dir,
        device=device,
    )


