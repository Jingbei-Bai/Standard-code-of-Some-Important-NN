import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import Legendre


torch.manual_seed(0)
np.random.seed(0)
torch.set_default_dtype(torch.float64)


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


def forcing_np(x):
    return (np.pi ** 2) * np.sin(np.pi * x)


def forcing_torch(x):
    return (np.pi ** 2) * torch.sin(np.pi * x)


def predict_u(model, x):

    return (1.0 - x ** 2) * model(x)


def build_reference_test_functions(p_order, n_quad):
    xi, w = np.polynomial.legendre.leggauss(n_quad)
    v = np.zeros((p_order, n_quad), dtype=np.float64)
    dv_dxi = np.zeros((p_order, n_quad), dtype=np.float64)
    for k in range(1, p_order + 1):
        vk = Legendre.basis(k + 1) - Legendre.basis(k - 1)
        dvk = vk.deriv()
        v[k - 1, :] = vk(xi)
        dv_dxi[k - 1, :] = dvk(xi)
    return xi, w, v, dv_dxi


def build_uniform_mesh(n_elem, x_lb=-1.0, x_rb=1.0):
    edges = np.linspace(x_lb, x_rb, n_elem + 1, dtype=np.float64)
    left = edges[:-1]
    right = edges[1:]
    centers = 0.5 * (left + right)
    half_sizes = 0.5 * (right - left)
    return edges, centers, half_sizes


def variational_residuals(model, centers, half_sizes, xi_t, w_t, v_t, dv_t):
    n_elem = centers.shape[0]
    n_quad = xi_t.numel()

    xq = centers.view(n_elem, 1) + half_sizes.view(n_elem, 1) * xi_t.view(1, n_quad)
    xq = xq.reshape(-1, 1).clone().detach().requires_grad_(True)

    u = predict_u(model, xq)
    u_x = torch.autograd.grad(
        u, xq, grad_outputs=torch.ones_like(u), create_graph=True
    )[0].reshape(n_elem, n_quad)
    f = forcing_torch(xq).reshape(n_elem, n_quad)

    weighted_u_x = u_x * w_t.view(1, n_quad)
    weighted_f = f * w_t.view(1, n_quad)



    lhs = torch.einsum("eq,kq->ek", weighted_u_x, dv_t)
    rhs = torch.einsum("e,eq,kq->ek", half_sizes, weighted_f, v_t)
    return lhs - rhs


def train_hp_vpinn_poisson(
    x_lb=-1.0,
    x_rb=1.0,
    n_elem=8,
    p_order=4,
    n_quad=20,
    hidden=64,
    n_layers=4,
    adam_epochs=4000,
    lbfgs_iters=500,
    lr=1e-3,
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

    if n_quad < p_order + 2:
        raise ValueError("n_quad should be at least p_order + 2 for stable integration.")

    _, centers_np, half_sizes_np = build_uniform_mesh(n_elem, x_lb=x_lb, x_rb=x_rb)
    xi, w, v, dv = build_reference_test_functions(p_order=p_order, n_quad=n_quad)

    centers_t = torch.tensor(centers_np, device=device)
    half_sizes_t = torch.tensor(half_sizes_np, device=device)
    xi_t = torch.tensor(xi, device=device)
    w_t = torch.tensor(w, device=device)
    v_t = torch.tensor(v, device=device)
    dv_t = torch.tensor(dv, device=device)

    model = MLP(in_dim=1, out_dim=1, hidden=hidden, n_layers=n_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []

    for epoch in range(1, adam_epochs + 1):
        model.train()
        optimizer.zero_grad()
        residuals = variational_residuals(
            model, centers_t, half_sizes_t, xi_t, w_t, v_t, dv_t
        )
        loss = torch.mean(residuals ** 2)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if epoch == 1 or epoch % 500 == 0:
            print(
                f"adam epoch {epoch}/{adam_epochs}, "
                f"variational loss={loss.item():.6e}"
            )

    if lbfgs_iters > 0:
        optimizer_lbfgs = optim.LBFGS(
            model.parameters(),
            max_iter=lbfgs_iters,
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        print("starting L-BFGS stage")

        def closure():
            optimizer_lbfgs.zero_grad()
            residuals = variational_residuals(
                model, centers_t, half_sizes_t, xi_t, w_t, v_t, dv_t
            )
            loss_val = torch.mean(residuals ** 2)
            loss_val.backward()
            return loss_val

        optimizer_lbfgs.step(closure)
        residuals = variational_residuals(
            model, centers_t, half_sizes_t, xi_t, w_t, v_t, dv_t
        )
        loss_history.append(torch.mean(residuals ** 2).item())

    x_eval = np.linspace(x_lb, x_rb, 1001, dtype=np.float64)[:, None]
    x_eval_t = torch.tensor(x_eval, device=device)
    model.eval()
    with torch.no_grad():
        u_pred = predict_u(model, x_eval_t).cpu().numpy().squeeze()
    u_ref = exact_solution_np(x_eval.squeeze())
    rel_l2 = np.linalg.norm(u_pred - u_ref) / (np.linalg.norm(u_ref) + 1e-12)

    x_grad_t = torch.tensor(x_eval, device=device, requires_grad=True)
    u_pred_t = predict_u(model, x_grad_t)
    du_pred = torch.autograd.grad(
        u_pred_t, x_grad_t, grad_outputs=torch.ones_like(u_pred_t), create_graph=False
    )[0].detach().cpu().numpy().squeeze()
    du_ref = np.pi * np.cos(np.pi * x_eval.squeeze())
    h1_semi = np.sqrt(np.trapezoid((du_pred - du_ref) ** 2, x_eval.squeeze()))

    print(f"relative L2 error={rel_l2:.6e}")
    print(f"H1 semi-norm error={h1_semi:.6e}")

    np.save(os.path.join(out_dir, "hp_vpinn_poisson_loss.npy"), np.array(loss_history))
    np.save(
        os.path.join(out_dir, "hp_vpinn_poisson_pred.npy"),
        np.stack([x_eval.squeeze(), u_pred, u_ref], axis=1),
    )
    torch.save(model.state_dict(), os.path.join(out_dir, "hp_vpinn_poisson_1d.pt"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(x_eval.squeeze(), u_ref, "k-", lw=2, label="exact")
    axes[0].plot(x_eval.squeeze(), u_pred, "r--", lw=2, label="hp-VPINN")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(x)")
    axes[0].set_title("1D Poisson: solution")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(x_eval.squeeze(), np.abs(u_pred - u_ref), "b-", lw=2, label="|error|")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("absolute error")
    axes[1].set_title("Pointwise error (log scale)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "hp_vpinn_poisson_1d_compare.png")
    plt.savefig(fig_path, dpi=180)
    print("saved figure to", fig_path)

    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, "m-", lw=1.5)
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel("variational loss")
    plt.title("hp-VPINN training loss")
    plt.grid(alpha=0.3)
    loss_fig = os.path.join(out_dir, "hp_vpinn_poisson_1d_loss.png")
    plt.tight_layout()
    plt.savefig(loss_fig, dpi=180)
    print("saved loss figure to", loss_fig)

    return rel_l2, h1_semi


if __name__ == "__main__":
    device = set_device()
    print("device:", device)
    train_hp_vpinn_poisson(
        x_lb=-1.0,
        x_rb=1.0,
        n_elem=8,
        p_order=4,
        n_quad=20,
        hidden=64,
        n_layers=4,
        adam_epochs=4000,
        lbfgs_iters=500,
        lr=1e-3,
        out_dir=None,
        device=device,
    )


