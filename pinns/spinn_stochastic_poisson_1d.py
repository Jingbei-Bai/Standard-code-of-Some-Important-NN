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

    def forward(self, x_xi):
        return self.net(x_xi)


def forcing_torch(x, xi):

    return (1.0 + 0.5 * xi) * (np.pi ** 2) * torch.sin(np.pi * x)


def exact_solution_np(x, xi):
    return (1.0 + 0.5 * xi) * np.sin(np.pi * x)


def sample_collocation(n_f, xi_lb=-1.0, xi_rb=1.0, x_lb=0.0, x_rb=1.0):
    x = np.random.uniform(x_lb, x_rb, size=(n_f, 1)).astype(np.float32)
    xi = np.random.uniform(xi_lb, xi_rb, size=(n_f, 1)).astype(np.float32)
    return np.hstack([x, xi]).astype(np.float32)


def sample_boundary(n_bc, xi_lb=-1.0, xi_rb=1.0, x_lb=0.0, x_rb=1.0):
    xi = np.random.uniform(xi_lb, xi_rb, size=(n_bc, 1)).astype(np.float32)
    xl = np.ones((n_bc, 1), dtype=np.float32) * x_lb
    xr = np.ones((n_bc, 1), dtype=np.float32) * x_rb
    xb_l = np.hstack([xl, xi]).astype(np.float32)
    xb_r = np.hstack([xr, xi]).astype(np.float32)
    return xb_l, xb_r


def pde_residual(model, x_xi):
    x_xi = x_xi.clone().detach().requires_grad_(True)
    u = model(x_xi)
    grads = torch.autograd.grad(
        u, x_xi, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_x = grads[:, 0:1]
    u_xx = torch.autograd.grad(
        u_x, x_xi, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0][:, 0:1]

    x = x_xi[:, 0:1]
    xi = x_xi[:, 1:2]
    f = forcing_torch(x, xi)
    return -u_xx - f


def evaluate_stochastic_solution(model, xi_test, x_eval, device):
    n_xi = xi_test.shape[0]
    nx = x_eval.shape[0]

    X = np.repeat(x_eval[None, :], n_xi, axis=0).astype(np.float32)
    XI = np.repeat(xi_test[:, None], nx, axis=1).astype(np.float32)
    inp = np.stack([X, XI], axis=2).reshape(-1, 2).astype(np.float32)

    model.eval()
    with torch.no_grad():
        u_pred = model(torch.tensor(inp, dtype=torch.float32, device=device)).cpu().numpy().reshape(n_xi, nx)
    u_ref = exact_solution_np(X, XI)

    rel_l2 = np.linalg.norm(u_pred - u_ref) / (np.linalg.norm(u_ref) + 1e-12)
    max_abs = np.max(np.abs(u_pred - u_ref))
    return u_pred, u_ref, rel_l2, max_abs


def train_spinn_stochastic_poisson_1d(
    xi_lb=-1.0,
    xi_rb=1.0,
    x_lb=0.0,
    x_rb=1.0,
    n_f=6000,
    n_bc=800,
    adam_epochs=4000,
    lbfgs_iters=300,
    lr=1e-3,
    lambda_bc=20.0,
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

    model = MLP(in_dim=2, out_dim=1, hidden=hidden, n_layers=n_layers).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    mse_f_history = []
    mse_bc_history = []

    for epoch in range(1, adam_epochs + 1):
        model.train()
        x_f_np = sample_collocation(n_f, xi_lb=xi_lb, xi_rb=xi_rb, x_lb=x_lb, x_rb=x_rb)
        xb_l_np, xb_r_np = sample_boundary(n_bc, xi_lb=xi_lb, xi_rb=xi_rb, x_lb=x_lb, x_rb=x_rb)

        x_f = torch.tensor(x_f_np, dtype=torch.float32, device=device)
        xb_l = torch.tensor(xb_l_np, dtype=torch.float32, device=device)
        xb_r = torch.tensor(xb_r_np, dtype=torch.float32, device=device)

        r = pde_residual(model, x_f)
        mse_f = torch.mean(r ** 2)
        mse_bc = 0.5 * (torch.mean(model(xb_l) ** 2) + torch.mean(model(xb_r) ** 2))
        loss = mse_f + lambda_bc * mse_bc

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_history.append(loss.item())
        mse_f_history.append(mse_f.item())
        mse_bc_history.append(mse_bc.item())

        if epoch == 1 or epoch % 500 == 0:
            print(
                f"adam epoch {epoch}/{adam_epochs}, loss={loss.item():.6e}, "
                f"mse_f={mse_f.item():.6e}, mse_bc={mse_bc.item():.6e}"
            )

    if lbfgs_iters > 0:
        x_f_np = sample_collocation(n_f, xi_lb=xi_lb, xi_rb=xi_rb, x_lb=x_lb, x_rb=x_rb)
        xb_l_np, xb_r_np = sample_boundary(n_bc, xi_lb=xi_lb, xi_rb=xi_rb, x_lb=x_lb, x_rb=x_rb)

        x_f = torch.tensor(x_f_np, dtype=torch.float32, device=device)
        xb_l = torch.tensor(xb_l_np, dtype=torch.float32, device=device)
        xb_r = torch.tensor(xb_r_np, dtype=torch.float32, device=device)

        opt_lbfgs = optim.LBFGS(
            model.parameters(),
            max_iter=lbfgs_iters,
            tolerance_grad=1e-10,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt_lbfgs.zero_grad()
            r = pde_residual(model, x_f)
            mse_f = torch.mean(r ** 2)
            mse_bc = 0.5 * (torch.mean(model(xb_l) ** 2) + torch.mean(model(xb_r) ** 2))
            loss_val = mse_f + lambda_bc * mse_bc
            loss_val.backward()
            return loss_val

        print("starting L-BFGS stage")
        opt_lbfgs.step(closure)

    xi_test = np.linspace(xi_lb, xi_rb, 101, dtype=np.float32)
    x_eval = np.linspace(x_lb, x_rb, 201, dtype=np.float32)
    u_pred, u_ref, rel_l2, max_abs = evaluate_stochastic_solution(
        model, xi_test=xi_test, x_eval=x_eval, device=device
    )
    print(f"relative L2 error={rel_l2:.6e}")
    print(f"max absolute error={max_abs:.6e}")


    mean_pred = np.mean(u_pred, axis=0)
    std_pred = np.std(u_pred, axis=0)

    mean_ref = np.sin(np.pi * x_eval)
    std_ref = (1.0 / np.sqrt(12.0)) * np.abs(np.sin(np.pi * x_eval))

    mean_rel_l2 = np.linalg.norm(mean_pred - mean_ref) / (np.linalg.norm(mean_ref) + 1e-12)
    std_rel_l2 = np.linalg.norm(std_pred - std_ref) / (np.linalg.norm(std_ref) + 1e-12)
    print(f"mean profile relative L2 error={mean_rel_l2:.6e}")
    print(f"std  profile relative L2 error={std_rel_l2:.6e}")

    np.save(os.path.join(out_dir, "spinn_stochastic_poisson_loss.npy"), np.array(loss_history))
    np.save(os.path.join(out_dir, "spinn_stochastic_poisson_mse_f.npy"), np.array(mse_f_history))
    np.save(os.path.join(out_dir, "spinn_stochastic_poisson_mse_bc.npy"), np.array(mse_bc_history))
    np.savez(
        os.path.join(out_dir, "spinn_stochastic_poisson_pred.npz"),
        xi_test=xi_test,
        x_eval=x_eval,
        u_pred=u_pred,
        u_ref=u_ref,
        mean_pred=mean_pred,
        mean_ref=mean_ref,
        std_pred=std_pred,
        std_ref=std_ref,
    )
    torch.save(model.state_dict(), os.path.join(out_dir, "spinn_stochastic_poisson_1d.pt"))

    plt.figure(figsize=(13, 8))
    plt.subplot(2, 2, 1)
    idx = [0, 25, 50, 75, 100]
    for i in idx:
        plt.plot(x_eval, u_ref[i], "-", lw=1.8, label=f"ref xi={xi_test[i]:.2f}")
        plt.plot(x_eval, u_pred[i], "--", lw=1.5, label=f"sPINN xi={xi_test[i]:.2f}")
    plt.xlabel("x")
    plt.ylabel("u(x,xi)")
    plt.title("Random realizations")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=8)

    plt.subplot(2, 2, 2)
    XX, XI = np.meshgrid(x_eval, xi_test)
    pcm = plt.pcolormesh(XX, XI, np.abs(u_pred - u_ref), shading="auto", cmap="magma")
    plt.colorbar(pcm)
    plt.xlabel("x")
    plt.ylabel("xi")
    plt.title("|prediction error| on (x,xi)")

    plt.subplot(2, 2, 3)
    plt.plot(x_eval, mean_ref, "k-", lw=2, label="exact mean")
    plt.plot(x_eval, mean_pred, "r--", lw=2, label="sPINN mean")
    plt.xlabel("x")
    plt.ylabel("E[u]")
    plt.title("Mean profile")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(x_eval, std_ref, "k-", lw=2, label="exact std")
    plt.plot(x_eval, std_pred, "b--", lw=2, label="sPINN std")
    plt.xlabel("x")
    plt.ylabel("Std[u]")
    plt.title("Standard deviation profile")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "spinn_stochastic_poisson_1d_compare.png")
    plt.savefig(fig_path, dpi=180)
    print("saved figure to", fig_path)

    plt.figure(figsize=(8, 4.8))
    plt.plot(loss_history, label="total", lw=1.5)
    plt.plot(mse_f_history, label="mse_f", lw=1.2)
    plt.plot(np.array(mse_bc_history) * lambda_bc, label="lambda_bc*mse_bc", lw=1.2)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss terms")
    plt.title("sPINN training history")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    loss_fig = os.path.join(out_dir, "spinn_stochastic_poisson_1d_loss.png")
    plt.savefig(loss_fig, dpi=180)
    print("saved loss figure to", loss_fig)
    return rel_l2, max_abs, mean_rel_l2, std_rel_l2


if __name__ == "__main__":
    device = set_device()
    print("device:", device)
    train_spinn_stochastic_poisson_1d(
        xi_lb=-1.0,
        xi_rb=1.0,
        x_lb=0.0,
        x_rb=1.0,
        n_f=6000,
        n_bc=800,
        adam_epochs=4000,
        lbfgs_iters=300,
        lr=1e-3,
        lambda_bc=20.0,
        hidden=96,
        n_layers=4,
        out_dir=None,
        device=device,
    )


