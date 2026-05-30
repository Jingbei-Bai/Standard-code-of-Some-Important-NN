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


def exact_solution_np(x):
    return np.sin(np.pi * x)


def forcing_torch(x):
    return (np.pi ** 2) * torch.sin(np.pi * x)


def forcing_np(x):
    return (np.pi ** 2) * np.sin(np.pi * x)


def gpinn_residuals(model, x_f):

    x_f = x_f.clone().detach().requires_grad_(True)
    u = model(x_f)
    u_x = torch.autograd.grad(
        u, x_f, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]

    f = forcing_torch(x_f)
    r = -u_xx - f
    r_x = torch.autograd.grad(
        r, x_f, grad_outputs=torch.ones_like(r), create_graph=True
    )[0]
    return r, r_x


def sample_interior(n_f, x_lb=-1.0, x_rb=1.0):
    return np.random.uniform(x_lb, x_rb, size=(n_f, 1)).astype(np.float32)


def train_gpinn_poisson_1d(
    x_lb=-1.0,
    x_rb=1.0,
    n_f=4000,
    adam_epochs=6000,
    lbfgs_iters=500,
    lr=1e-3,
    lambda_g=0.1,
    lambda_bc=10.0,
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

    model = MLP(in_dim=1, out_dim=1, hidden=hidden, n_layers=n_layers).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    x_bc_np = np.array([[x_lb], [x_rb]], dtype=np.float32)
    u_bc_np = np.array([[0.0], [0.0]], dtype=np.float32)
    x_bc = torch.tensor(x_bc_np, dtype=torch.float32, device=device)
    u_bc = torch.tensor(u_bc_np, dtype=torch.float32, device=device)

    loss_history = []
    mse_r_history = []
    mse_rx_history = []
    mse_bc_history = []

    for epoch in range(1, adam_epochs + 1):
        model.train()
        x_f_np = sample_interior(n_f, x_lb=x_lb, x_rb=x_rb)
        x_f = torch.tensor(x_f_np, dtype=torch.float32, device=device)

        r, r_x = gpinn_residuals(model, x_f)
        mse_r = torch.mean(r ** 2)
        mse_rx = torch.mean(r_x ** 2)
        mse_bc = torch.mean((model(x_bc) - u_bc) ** 2)

        loss = mse_r + lambda_g * mse_rx + lambda_bc * mse_bc
        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_history.append(loss.item())
        mse_r_history.append(mse_r.item())
        mse_rx_history.append(mse_rx.item())
        mse_bc_history.append(mse_bc.item())

        if epoch == 1 or epoch % 500 == 0:
            print(
                f"adam epoch {epoch}/{adam_epochs}, loss={loss.item():.6e}, "
                f"mse_r={mse_r.item():.6e}, mse_rx={mse_rx.item():.6e}, mse_bc={mse_bc.item():.6e}"
            )

    if lbfgs_iters > 0:
        x_f_lb_np = sample_interior(n_f, x_lb=x_lb, x_rb=x_rb)
        x_f_lb = torch.tensor(x_f_lb_np, dtype=torch.float32, device=device)

        opt_lbfgs = optim.LBFGS(
            model.parameters(),
            max_iter=lbfgs_iters,
            tolerance_grad=1e-10,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt_lbfgs.zero_grad()
            r, r_x = gpinn_residuals(model, x_f_lb)
            mse_r = torch.mean(r ** 2)
            mse_rx = torch.mean(r_x ** 2)
            mse_bc = torch.mean((model(x_bc) - u_bc) ** 2)
            loss_val = mse_r + lambda_g * mse_rx + lambda_bc * mse_bc
            loss_val.backward()
            return loss_val

        print("starting L-BFGS stage")
        opt_lbfgs.step(closure)

    x_eval = np.linspace(x_lb, x_rb, 1001, dtype=np.float32)[:, None]
    x_eval_t = torch.tensor(x_eval, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        u_pred = model(x_eval_t).cpu().numpy().squeeze()

    u_ref = exact_solution_np(x_eval.squeeze())
    rel_l2 = np.linalg.norm(u_pred - u_ref) / (np.linalg.norm(u_ref) + 1e-12)
    max_abs = np.max(np.abs(u_pred - u_ref))
    print(f"relative L2 error={rel_l2:.6e}")
    print(f"max absolute error={max_abs:.6e}")

    np.save(os.path.join(out_dir, "gpinn_poisson_loss.npy"), np.array(loss_history))
    np.save(os.path.join(out_dir, "gpinn_poisson_mse_r.npy"), np.array(mse_r_history))
    np.save(os.path.join(out_dir, "gpinn_poisson_mse_rx.npy"), np.array(mse_rx_history))
    np.save(os.path.join(out_dir, "gpinn_poisson_mse_bc.npy"), np.array(mse_bc_history))
    np.save(
        os.path.join(out_dir, "gpinn_poisson_pred.npy"),
        np.stack([x_eval.squeeze(), u_pred, u_ref], axis=1),
    )
    torch.save(model.state_dict(), os.path.join(out_dir, "gpinn_poisson_1d.pt"))

    plt.figure(figsize=(12, 4.2))
    plt.subplot(1, 2, 1)
    plt.plot(x_eval.squeeze(), u_ref, "k-", lw=2, label="exact")
    plt.plot(x_eval.squeeze(), u_pred, "r--", lw=2, label="gPINN")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("1D Poisson solution")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x_eval.squeeze(), np.abs(u_pred - u_ref), "b-", lw=2)
    plt.yscale("log")
    plt.xlabel("x")
    plt.ylabel("|error|")
    plt.title("Pointwise absolute error")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "gpinn_poisson_1d_compare.png")
    plt.savefig(fig_path, dpi=180)
    print("saved figure to", fig_path)

    plt.figure(figsize=(7, 4.5))
    plt.plot(loss_history, label="total loss", lw=1.5)
    plt.plot(mse_r_history, label="mse_r", lw=1.1)
    plt.plot(np.array(mse_rx_history) * lambda_g, label="lambda_g*mse_rx", lw=1.1)
    plt.plot(np.array(mse_bc_history) * lambda_bc, label="lambda_bc*mse_bc", lw=1.1)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss terms")
    plt.title("gPINN training history")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    loss_fig = os.path.join(out_dir, "gpinn_poisson_1d_loss.png")
    plt.savefig(loss_fig, dpi=180)
    print("saved loss figure to", loss_fig)

    return rel_l2, max_abs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gPINN for 1D Poisson equation")
    parser.add_argument("--x_lb", type=float, default=-1.0)
    parser.add_argument("--x_rb", type=float, default=1.0)
    parser.add_argument("--n_f", type=int, default=4000)
    parser.add_argument("--adam_epochs", type=int, default=6000)
    parser.add_argument("--lbfgs_iters", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_g", type=float, default=0.1, help="weight of residual-gradient term")
    parser.add_argument("--lambda_bc", type=float, default=10.0, help="weight of boundary loss")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    device = set_device()
    print("device:", device)
    train_gpinn_poisson_1d(
        x_lb=args.x_lb,
        x_rb=args.x_rb,
        n_f=args.n_f,
        adam_epochs=args.adam_epochs,
        lbfgs_iters=args.lbfgs_iters,
        lr=args.lr,
        lambda_g=args.lambda_g,
        lambda_bc=args.lambda_bc,
        hidden=args.hidden,
        n_layers=args.n_layers,
        out_dir=args.out_dir,
        device=device,
    )


