import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def set_device():
    return torch.device("cuda")


class MLP(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, hidden=50, n_layers=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.Tanh())
        for _ in range(n_layers-1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def train_pinn_bvp(nu=1e-2, x_lb=-1.0, x_rb=1.0, N_f=2000, epochs=5000, lr=1e-3, out_dir=None, device=None):
    if device is None:
        device = set_device()
    if out_dir is None:
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ode_solvers_outputs', 'pinns'))
    os.makedirs(out_dir, exist_ok=True)

    model = MLP(in_dim=1, out_dim=1, hidden=50, n_layers=3).to(device)

    x_f = np.random.uniform(x_lb, x_rb, size=(N_f,1))
    x_f_t = torch.tensor(x_f, dtype=torch.float32, device=device, requires_grad=True)

    x_bc = np.array([[x_lb],[x_rb]], dtype=np.float32)
    u_bc = np.array([[1.0],[0.0]], dtype=np.float32)
    x_bc_t = torch.tensor(x_bc, dtype=torch.float32, device=device)
    u_bc_t = torch.tensor(u_bc, dtype=torch.float32, device=device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()

        u_f = model(x_f_t)
        u_x = torch.autograd.grad(u_f, x_f_t, grad_outputs=torch.ones_like(u_f), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_f_t, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        rhs = torch.exp(x_f_t)
        f_res = nu * u_xx - u_f - rhs
        mse_f = torch.mean(f_res**2)

        u_bc_pred = model(x_bc_t)
        mse_bc = torch.mean((u_bc_pred - u_bc_t)**2)

        loss = mse_f + mse_bc
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0 or epoch == 1:
            print(f'epoch {epoch}/{epochs}, loss={loss.item():.6e}, mse_f={mse_f.item():.6e}, mse_bc={mse_bc.item():.6e}')

    model_path = os.path.join(out_dir, 'pinn_bvp.pt')
    torch.save(model.state_dict(), model_path)

    def analytic_solution(nu, x):
        s = 1.0 / np.sqrt(nu)
        A = 1.0 / (nu - 1.0)
        e_sx = np.exp(s * x)
        e_minus_sx = np.exp(-s * x)
        e_x = np.exp(x)
        b0 = 1.0 - A * np.exp(-1.0)
        b1 = - A * np.exp(1.0)
        M = np.array([[np.exp(-s), np.exp(s)], [np.exp(s), np.exp(-s)]], dtype=np.float64)
        rhs = np.array([b0, b1], dtype=np.float64)
        C1, C2 = np.linalg.solve(M, rhs)
        u = C1 * e_sx + C2 * e_minus_sx + A * e_x
        return u

    x_plot = np.linspace(x_lb, x_rb, 501)[:,None]
    x_plot_t = torch.tensor(x_plot, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        u_pred = model(x_plot_t).cpu().numpy().squeeze()
    u_ref = analytic_solution(nu, x_plot.squeeze())

    plt.figure(figsize=(8,4))
    plt.plot(x_plot.squeeze(), u_ref, '-', label='analytic')
    plt.plot(x_plot.squeeze(), u_pred, '--', label='pinn')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    fig_path = os.path.join(out_dir, 'pinn_bvp_compare.png')
    plt.savefig(fig_path)
    print('saved', fig_path)


if __name__ == '__main__':
    train_pinn_bvp(nu=1e-2, N_f=3000, epochs=5000)


