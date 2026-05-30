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


class MLP(nn.Module):
    def __init__(self, inp, out, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, out)
        )
    def forward(self,x):
        return self.net(x)


def gen_harmonic(n=200):
    t = np.linspace(0,2*math.pi, n)
    q = np.cos(t)
    p = -np.sin(t)
    z = np.stack([q,p],axis=1)
    dz = np.stack([-p,-q],axis=1)
    return z.astype(np.float32), dz.astype(np.float32)

def train_hamiltonian(device, out_dir):
    z_np, dz_np = gen_harmonic(400)
    X = torch.tensor(z_np, device=device, dtype=torch.float32)
    X = X.clone().detach().requires_grad_(True)
    Y = torch.tensor(dz_np, device=device, dtype=torch.float32)
    model = MLP(2,1,128).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    Jinv = torch.tensor([[0.0,1.0],[-1.0,0.0]], device=device)
    model.train()
    for epoch in range(600):
        opt.zero_grad()
        H = model(X).squeeze()
        if not X.requires_grad:
            X = X.clone().detach().requires_grad_(True)
        if not H.requires_grad:
            H = model(X).squeeze()
        grads = torch.autograd.grad(H.sum(), X, create_graph=True)[0]
        f = grads.matmul(Jinv.t())
        loss = ((f - Y)**2).mean()
        loss.backward()
        opt.step()

    H = model(X).squeeze()
    grads = torch.autograd.grad(H.sum(), X, create_graph=False)[0]
    f = grads.matmul(Jinv.t()).detach().cpu().numpy()
    plt.figure(figsize=(6,3))
    plt.plot(dz_np[:,0], label='dq_true')
    plt.plot(f[:,0], '--', label='dq_pred')
    plt.legend()
    path = os.path.join(out_dir, 'hamiltonian_harmonic.png')
    plt.savefig(path)
    print('saved', path)


def gen_pendulum(n=400):
    t = np.linspace(0,10,n)

    q = np.cos(t)
    qdot = -np.sin(t)
    qdd = -q
    X = np.stack([q,qdot],axis=1).astype(np.float32)
    Y = np.stack([qdot,qdd],axis=1).astype(np.float32)
    return X,Y

def train_lagrangian(device, out_dir):
    X_np,Y_np = gen_pendulum(500)
    X = torch.tensor(X_np, device=device, requires_grad=True)
    Y = torch.tensor(Y_np, device=device)
    Vnet = MLP(1,1,128).to(device)
    opt = optim.Adam(Vnet.parameters(), lr=1e-3)
    for epoch in range(500):
        opt.zero_grad()
        q = X[:,0:1]
        V = Vnet(q).squeeze()
        dV_dq = torch.autograd.grad(V.sum(), q, create_graph=True)[0].squeeze()
        qdd_pred = -dV_dq
        loss = ((qdd_pred - Y[:,1])**2).mean()
        loss.backward()
        opt.step()

    q = torch.tensor(X_np[:,0:1], device=device, requires_grad=True)
    V = Vnet(q).squeeze()
    dV_dq = torch.autograd.grad(V.sum(), q, create_graph=False)[0].squeeze().detach().cpu().numpy()
    qdd_pred = -dV_dq
    plt.figure(figsize=(6,3))
    plt.plot(Y_np[:,1], label='qdd_true')
    plt.plot(qdd_pred, '--', label='qdd_pred')
    plt.legend()
    path = os.path.join(out_dir, 'lagrangian_pendulum.png')
    plt.savefig(path)
    print('saved', path)


def gen_lv(n=400,alpha=1.5,beta=1.0,delta=1.0,gamma=3.0):
    t = np.linspace(0,10,n)

    def f(x):
        a,b = x
        return np.array([alpha*a - beta*a*b, delta*a*b - gamma*b])
    dt = t[1]-t[0]
    x = np.array([1.0,1.0])
    traj=[]
    for ti in t:
        traj.append(x.copy())
        k1 = f(x)
        k2 = f(x+0.5*dt*k1)
        k3 = f(x+0.5*dt*k2)
        k4 = f(x+dt*k3)
        x = x + dt/6*(k1+2*k2+2*k3+k4)
    traj = np.stack(traj)
    dz = np.array([f(xx) for xx in traj]).astype(np.float32)
    return traj.astype(np.float32), dz


def gen_lv_augmented(nsamples=256, nsteps=400, T=10.0, alpha=1.5, beta=1.0, delta=1.0, gamma=3.0):
    t = np.linspace(0, T, nsteps)
    dt = t[1] - t[0]


    def f(x, a=alpha, b=beta, d=delta, g=gamma):

        x = np.asarray(x, dtype=np.float64)

        x = np.clip(x, 1e-8, 1e8)
        return np.array([a * x[0] - b * x[0] * x[1], d * x[0] * x[1] - g * x[1]], dtype=np.float64)

    data = []
    derivs = []
    for s in range(nsamples):

        x = np.abs(np.array([1.0, 1.0], dtype=np.float64) + 0.5 * np.random.randn(2))
        traj = []
        trajd = []
        diverged = False
        for ti in t:
            traj.append(x.copy())
            k1 = f(x)
            k2 = f(x + 0.5 * dt * k1)
            k3 = f(x + 0.5 * dt * k2)
            k4 = f(x + dt * k3)
            x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

            x = np.clip(x, 1e-8, 1e8)
            fx = f(x)

            if not np.isfinite(x).all() or not np.isfinite(fx).all():
                diverged = True
                break
            trajd.append(fx)
        if diverged:


            continue
        data.append(np.stack(traj).astype(np.float32))
        derivs.append(np.stack(trajd).astype(np.float32))


    if len(data) == 0:
        x = np.array([1.0, 1.0], dtype=np.float64)
        traj = []
        trajd = []
        for ti in t:
            traj.append(x.copy())
            k1 = f(x)
            k2 = f(x + 0.5 * dt * k1)
            k3 = f(x + 0.5 * dt * k2)
            k4 = f(x + dt * k3)
            x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            x = np.clip(x, 1e-8, 1e8)
            trajd.append(f(x))
        data.append(np.stack(traj).astype(np.float32))
        derivs.append(np.stack(trajd).astype(np.float32))
    data = np.concatenate(data, axis=0)
    derivs = np.concatenate(derivs, axis=0)
    return data, derivs

def train_poisson(device, out_dir):

    X_np, Y_np = gen_lv(n=400)
    X = torch.tensor(X_np, device=device, requires_grad=True)
    Y = torch.tensor(Y_np, device=device)
    Hnet = MLP(2,1,128).to(device)

    Anet = MLP(2,1,64).to(device)
    opt = optim.Adam(list(Hnet.parameters())+list(Anet.parameters()), lr=1e-3)
    for epoch in range(2500):
        opt.zero_grad()
        H = Hnet(X).squeeze()
        grads = torch.autograd.grad(H.sum(), X, create_graph=True)[0]
        a = Anet(X).squeeze()
        L = torch.stack([torch.stack([torch.zeros_like(a), a], dim=1), torch.stack([-a, torch.zeros_like(a)], dim=1)], dim=1)
        f = torch.einsum('bij,bj->bi', L, grads)
        loss = ((f - Y)**2).mean()
        loss.backward()
        opt.step()

    H_eval = Hnet(X).squeeze()
    grads = torch.autograd.grad(H_eval.sum(), X, create_graph=False)[0]
    a = Anet(X).squeeze()
    L = torch.stack([torch.stack([torch.zeros_like(a), a], dim=1), torch.stack([-a, torch.zeros_like(a)], dim=1)], dim=1)
    f = torch.einsum('bij,bj->bi', L, grads).detach().cpu().numpy()

    plt.figure(figsize=(6,3))
    plt.plot(Y[:,0].cpu().numpy(), label='dx_true')
    plt.plot(f[:,0], '--', label='dx_pred')
    plt.legend()
    path = os.path.join(out_dir, 'poisson_lv.png')
    plt.savefig(path)
    print('saved', path)




def gen_damped(n=400,gamma=0.1):
    t = np.linspace(0,20,n)
    q = np.exp(-gamma*t)*np.cos(t)
    p = -np.exp(-gamma*t)*(np.sin(t)+gamma*np.cos(t))
    z = np.stack([q,p],axis=1)
    dz = np.stack([p, -q - 2*gamma*p],axis=1)
    return z.astype(np.float32), dz.astype(np.float32)


def gen_damped_augmented(nsamples=256, nsteps=400, T=20.0):
    t = np.linspace(0, T, nsteps)
    data = []
    derivs = []
    for s in range(nsamples):
        gamma = np.random.uniform(0.05, 0.2)
        q = np.random.uniform(-1.0, 1.0)
        p = np.random.uniform(-1.0, 1.0)
        traj = []
        trajd = []
        dt = t[1] - t[0]
        for ti in t:
            traj.append(np.array([q, p]))
            dq = p
            dp = -q - 2 * gamma * p
            trajd.append(np.array([dq, dp]))
            q = q + dt * dq
            p = p + dt * dp
        data.append(np.stack(traj).astype(np.float32))
        derivs.append(np.stack(trajd).astype(np.float32))
    data = np.concatenate(data, axis=0)
    derivs = np.concatenate(derivs, axis=0)
    return data, derivs

def train_generic(device, out_dir):

    Z_np, DZ_np = gen_damped_augmented(nsamples=256, nsteps=400, T=20.0)
    X = torch.tensor(Z_np, device=device, requires_grad=True)
    Y = torch.tensor(DZ_np, device=device)
    Enet = MLP(2,1,128).to(device)
    Snet = MLP(2,1,128).to(device)

    Lnet = MLP(2,1,64).to(device)

    k = 2
    Bnet = MLP(2,2*k,128).to(device)
    opt = optim.Adam(list(Enet.parameters())+list(Snet.parameters())+list(Lnet.parameters())+list(Bnet.parameters()), lr=1e-3)
    for epoch in range(5000):
        opt.zero_grad()
        E = Enet(X).squeeze()
        S = Snet(X).squeeze()
        gradE = torch.autograd.grad(E.sum(), X, create_graph=True)[0]
        gradS = torch.autograd.grad(S.sum(), X, create_graph=True)[0]
        a = Lnet(X).squeeze()
        L = torch.stack([torch.stack([torch.zeros_like(a), a], dim=1), torch.stack([-a, torch.zeros_like(a)], dim=1)], dim=1)
        Bflat = Bnet(X)
        B = Bflat.view(-1,2,k)
        M = torch.matmul(B, B.transpose(1,2))
        f_rev = torch.einsum('bij,bj->bi', L, gradE)
        f_irr = torch.einsum('bij,bj->bi', M, gradS)
        f = f_rev + f_irr
        loss = ((f - Y)**2).mean()
        loss.backward()
        opt.step()

    E_eval = Enet(X).squeeze()
    gradE = torch.autograd.grad(E_eval.sum(), X, create_graph=False)[0]
    a = Lnet(X).squeeze()
    L = torch.stack([torch.stack([torch.zeros_like(a), a], dim=1), torch.stack([-a, torch.zeros_like(a)], dim=1)], dim=1)
    f_rev = torch.einsum('bij,bj->bi', L, gradE).detach().cpu().numpy()
    plt.figure(figsize=(6,3))
    plt.plot(DZ_np[:,0], label='dq_true')
    plt.plot(f_rev[:,0], '--', label='dq_rev')
    plt.legend()
    path = os.path.join(out_dir, 'generic_damped.png')
    plt.savefig(path)
    print('saved', path)

if __name__=='__main__':
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'ode_solvers_outputs')
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    device = set_device()
    print('device', device)
    train_hamiltonian(device, out_dir)
    train_lagrangian(device, out_dir)
    train_poisson(device, out_dir)
    train_generic(device, out_dir)


