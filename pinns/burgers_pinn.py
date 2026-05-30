import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

def set_device():
    return torch.device("cuda")

class PINN(nn.Module):
    def __init__(self, layers=None):
        if layers is None:
            layers = [2,50,50,50,50,50,1]
        super().__init__()
        acts = []
        for i in range(len(layers)-2):
            acts.append(nn.Linear(layers[i], layers[i+1]))
            acts.append(nn.Tanh())
        acts.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*acts)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, tx):
        return self.net(tx)

def burgers_initial(x):
    return -np.sin(np.pi * x)

def solve_burgers_reference(nu, T, x_lb, x_rb, Nx=256):
    x = np.linspace(x_lb, x_rb, Nx)
    dx = x[1] - x[0]
    u0 = burgers_initial(x)
    max_u0 = np.max(np.abs(u0)) + 1e-8
    cfl_conv = 0.2
    cfl_diff = 0.1
    dt_conv = cfl_conv * dx / max_u0
    dt_diff = cfl_diff * dx * dx / max(nu, 1e-12)
    dt = min(dt_conv, dt_diff, T/2000.0)
    Nt = int(np.ceil(T / dt)) + 1
    dt = T / (Nt - 1)
    U = np.zeros((Nt, Nx), dtype=np.float64)
    U[0] = u0
    r = nu * dt / (dx * dx)
    Nint = Nx - 2
    if Nint <= 0:
        raise ValueError('Nx too small')
    diag = (1.0 + 2.0 * r) * np.ones(Nint)
    off = -r * np.ones(Nint - 1)
    A = np.diag(diag) + np.diag(off, -1) + np.diag(off, 1)
    for n in range(0, Nt-1):
        u_n = U[n].copy()
        f = 0.5 * u_n * u_n
        alpha = max(1e-6, np.max(np.abs(u_n)))
        F_half = 0.5*(f[:-1] + f[1:]) - 0.5 * alpha * (u_n[1:] - u_n[:-1])
        conv_term = np.zeros_like(u_n)
        conv_term[1:-1] = (F_half[1:] - F_half[:-1]) / dx
        rhs = u_n - dt * conv_term
        b = rhs[1:-1]
        u_int = np.linalg.solve(A, b)
        u_np1 = np.zeros_like(u_n)
        u_np1[0] = 0.0
        u_np1[-1] = 0.0
        u_np1[1:-1] = u_int
        if not np.isfinite(u_np1).all():
            u_np1 = np.clip(u_np1, -1e6, 1e6)
        U[n+1] = u_np1
    t = np.linspace(0, T, Nt)
    return t, x, U

def train_burgers(device=None, out_dir=None,
                  nu=0.01 / math.pi, T=1.0, x_lb=-1.0, x_rb=1.0,
                  N_f=10000, N_u=100, N_b=100, epochs=30000, lr=1e-4):
    if device is None:
        device = set_device()
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), '..', 'ode_solvers_outputs', 'pinns')
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    model = PINN().to(device)
    t_f = np.random.rand(N_f, 1) * T
    x_f = np.random.rand(N_f, 1) * (x_rb - x_lb) + x_lb
    x_u = np.linspace(x_lb, x_rb, N_u)[:, None]
    t_u = np.zeros_like(x_u)
    u0 = burgers_initial(x_u)
    t_b = np.random.rand(N_b, 1) * T
    x_b_left = np.ones((N_b,1)) * x_lb
    x_b_right = np.ones((N_b,1)) * x_rb
    tf = torch.tensor(t_f, dtype=torch.float32, device=device, requires_grad=True)
    xf = torch.tensor(x_f, dtype=torch.float32, device=device, requires_grad=True)
    tu = torch.tensor(t_u, dtype=torch.float32, device=device)
    xu = torch.tensor(x_u, dtype=torch.float32, device=device)
    u0_t = torch.tensor(u0, dtype=torch.float32, device=device)
    tb = torch.tensor(t_b, dtype=torch.float32, device=device)
    xb_l = torch.tensor(x_b_left, dtype=torch.float32, device=device)
    xb_r = torch.tensor(x_b_right, dtype=torch.float32, device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    def physics_residual(t, x):
        tx = torch.cat([t, x], dim=1)
        tx.requires_grad_(True)
        u = model(tx)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        f = u_t + u * u_x - nu * u_xx
        return f
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        f_pred = physics_residual(tf, xf)
        mse_f = torch.mean(f_pred**2)
        tu_in = torch.zeros_like(xu)
        tx_u = torch.cat([tu_in, xu], dim=1)
        u_pred_u = model(tx_u)
        mse_u = torch.mean((u_pred_u - u0_t)**2)
        tx_b_l = torch.cat([tb, xb_l], dim=1)
        tx_b_r = torch.cat([tb, xb_r], dim=1)
        u_b_l = model(tx_b_l)
        u_b_r = model(tx_b_r)
        mse_b = torch.mean(u_b_l**2) + torch.mean(u_b_r**2)
        loss = mse_f + mse_u + mse_b
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0 or epoch == 1:
            print(f'epoch {epoch}/{epochs}, loss={loss.item():.6e}, mse_f={mse_f.item():.6e}, mse_u={mse_u.item():.6e}, mse_b={mse_b.item():.6e}')
    model_path = os.path.join(out_dir, 'burgers_pinn.pt')
    torch.save(model.state_dict(), model_path)
    print('saved model to', model_path)
    model.eval()
    t_ref, x_ref, U_ref = solve_burgers_reference(nu=nu, T=T, x_lb=x_lb, x_rb=x_rb, Nx=256)
    Nt_ref = t_ref.size
    Nx_ref = x_ref.size
    U_pred = np.zeros((Nt_ref, Nx_ref), dtype=np.float32)
    with torch.no_grad():
        for i in range(Nt_ref):
            tt = t_ref[i]
            tt_arr = np.ones_like(x_ref) * tt
            tx = torch.tensor(np.stack([tt_arr, x_ref], axis=1), dtype=torch.float32, device=device)
            u_pred = model(tx).cpu().numpy().squeeze()
            U_pred[i] = u_pred
    Nt_show = 5
    ts_show_idx = np.linspace(0, Nt_ref-1, Nt_show, dtype=int)
    plt.figure(figsize=(8,6))
    for i in ts_show_idx:
        plt.plot(x_ref, U_ref[i], '-', label=f'ref t={t_ref[i]:.2f}')
        plt.plot(x_ref, U_pred[i], '--', label=f'PINN t={t_ref[i]:.2f}')
    plt.xlabel('x')
    plt.ylabel('u(t,x)')
    plt.title('Burgers PINN vs reference at several times')
    plt.legend()
    fig_path = os.path.join(out_dir, 'burgers_pinn_slices.png')
    plt.savefig(fig_path)
    print('saved plot to', fig_path)
    vmin = min(U_ref.min(), U_pred.min())
    vmax = max(U_ref.max(), U_pred.max())
    T_grid, X_grid = np.meshgrid(x_ref, t_ref)
    plt.figure(figsize=(12,8))
    plt.subplot(2,2,1)
    plt.title('Reference (t-x)')
    plt.pcolormesh(X_grid, T_grid, U_ref, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.subplot(2,2,2)
    plt.title('PINN prediction (t-x)')
    plt.pcolormesh(X_grid, T_grid, U_pred, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.subplot(2,2,3)
    plt.title('Difference (PINN - ref)')
    plt.pcolormesh(X_grid, T_grid, U_pred - U_ref, shading='auto', cmap='bwr')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.tight_layout()
    fig2_path = os.path.join(out_dir, 'burgers_pinn_compare_2d.png')
    plt.savefig(fig2_path)
    print('saved 2D comparison to', fig2_path)

if __name__ == '__main__':
    device = set_device()
    print('device', device)
    train_burgers(device=device, epochs=30000)


