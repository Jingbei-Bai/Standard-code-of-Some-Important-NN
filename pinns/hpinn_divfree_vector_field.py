import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


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

    def forward(self, x):
        return self.net(x)


def true_field_torch(x, a):
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    y = x1 * x2
    exp_term = torch.exp(-a * y)
    sin_term = torch.sin(y)
    cos_term = torch.cos(y)
    f1 = exp_term * (a * x1 * sin_term - x1 * cos_term)
    f2 = exp_term * (x2 * cos_term - a * x2 * sin_term)
    return f1, f2


def true_field_np(x1, x2, a):
    y = x1 * x2
    exp_term = np.exp(-a * y)
    sin_term = np.sin(y)
    cos_term = np.cos(y)
    f1 = exp_term * (a * x1 * sin_term - x1 * cos_term)
    f2 = exp_term * (x2 * cos_term - a * x2 * sin_term)
    return f1, f2


def hpinn_field(model, x):


    g = model(x)
    grad_g = torch.autograd.grad(
        g, x, grad_outputs=torch.ones_like(g), create_graph=True
    )[0]
    f1 = grad_g[:, 1:2]
    f2 = -grad_g[:, 0:1]
    return f1, f2


def sample_points(n_samples, lb, rb):
    x = np.random.uniform(lb, rb, size=(n_samples, 2)).astype(np.float32)
    return x


def quiver_display_vectors(u, v, clip_percentile=95.0, eps=1e-12):

    mag = np.sqrt(u ** 2 + v ** 2)
    mag_clip = np.percentile(mag, clip_percentile)
    mag_clip = max(float(mag_clip), 1e-8)
    rel = np.minimum(mag / mag_clip, 1.0)
    u_disp = (u / (mag + eps)) * rel
    v_disp = (v / (mag + eps)) * rel
    return u_disp, v_disp, mag


def pretty_quiver_panel(ax, X, Y, U, V, lb, rb, title, mag_norm):

    mag = np.sqrt(U ** 2 + V ** 2)
    im = ax.pcolormesh(
        X, Y, mag, shading="auto", cmap="Greys", norm=mag_norm, alpha=0.28
    )

    n_eval = X.shape[0]
    step = max(1, n_eval // 18)
    xq = X[::step, ::step]
    yq = Y[::step, ::step]
    uq = U[::step, ::step]
    vq = V[::step, ::step]
    mq = np.sqrt(uq ** 2 + vq ** 2)

    clip_val = np.percentile(mq, 92.0)
    clip_val = max(float(clip_val), 1e-8)
    rel = np.clip(mq / clip_val, 0.0, 1.0)

    len_scale = 0.25 + 0.75 * np.sqrt(rel)
    uq_disp = (uq / (mq + 1e-12)) * len_scale
    vq_disp = (vq / (mq + 1e-12)) * len_scale

    q = ax.quiver(
        xq,
        yq,
        uq_disp,
        vq_disp,
        mq,
        cmap="turbo",
        norm=mag_norm,
        angles="xy",
        scale_units="xy",
        scale=11.0,
        width=0.0042,
        pivot="mid",
        headwidth=4.2,
        headlength=5.0,
        headaxislength=4.5,
        linewidths=0.2,
    )

    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(lb, rb)
    ax.set_ylim(lb, rb)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15, linewidth=0.6)
    return q, im


def train_hpinn_divfree(
    a=2.0,
    lb=-1.0,
    rb=1.0,
    n_train=12000,
    batch_size=1024,
    adam_epochs=3000,
    lbfgs_iters=300,
    lr=1e-3,
    n_eval=121,
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

    model = MLP(in_dim=2, out_dim=1, hidden=128, n_layers=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    x_train_np = sample_points(n_train, lb=lb, rb=rb)
    x_train = torch.tensor(x_train_np, dtype=torch.float32, device=device)

    loss_history = []
    n_batches = int(np.ceil(n_train / batch_size))

    for epoch in range(1, adam_epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0

        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            xb = x_train[idx].clone().detach().requires_grad_(True)
            f1_true, f2_true = true_field_torch(xb, a=a)
            f1_pred, f2_pred = hpinn_field(model, xb)
            loss = torch.mean((f1_pred - f1_true) ** 2) + torch.mean((f2_pred - f2_true) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= max(1, n_batches)
        loss_history.append(epoch_loss)

        if epoch == 1 or epoch % 200 == 0:
            print(f"adam epoch {epoch}/{adam_epochs}, loss={epoch_loss:.6e}")

    if lbfgs_iters > 0:
        x_lbfgs = x_train.clone().detach().requires_grad_(True)
        opt_lbfgs = optim.LBFGS(
            model.parameters(),
            max_iter=lbfgs_iters,
            tolerance_grad=1e-10,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt_lbfgs.zero_grad()
            f1_true, f2_true = true_field_torch(x_lbfgs, a=a)
            f1_pred, f2_pred = hpinn_field(model, x_lbfgs)
            loss_val = torch.mean((f1_pred - f1_true) ** 2) + torch.mean((f2_pred - f2_true) ** 2)
            loss_val.backward()
            return loss_val

        print("starting L-BFGS stage")
        opt_lbfgs.step(closure)


    x1 = np.linspace(lb, rb, n_eval, dtype=np.float32)
    x2 = np.linspace(lb, rb, n_eval, dtype=np.float32)
    X1, X2 = np.meshgrid(x1, x2)
    X_eval_np = np.stack([X1.ravel(), X2.ravel()], axis=1).astype(np.float32)
    X_eval = torch.tensor(X_eval_np, dtype=torch.float32, device=device, requires_grad=True)

    model.eval()
    f1_true_t, f2_true_t = true_field_torch(X_eval, a=a)
    f1_pred_t, f2_pred_t = hpinn_field(model, X_eval)

    vec_true = torch.cat([f1_true_t, f2_true_t], dim=1)
    vec_pred = torch.cat([f1_pred_t, f2_pred_t], dim=1)
    rel_l2 = (
        torch.linalg.norm(vec_pred - vec_true) / (torch.linalg.norm(vec_true) + 1e-12)
    ).item()

    df1_dx1 = torch.autograd.grad(
        f1_pred_t,
        X_eval,
        grad_outputs=torch.ones_like(f1_pred_t),
        retain_graph=True,
        create_graph=False,
    )[0][:, 0:1]
    df2_dx2 = torch.autograd.grad(
        f2_pred_t,
        X_eval,
        grad_outputs=torch.ones_like(f2_pred_t),
        create_graph=False,
    )[0][:, 1:2]
    div_pred_t = df1_dx1 + df2_dx2

    f1_true = f1_true_t.detach().cpu().numpy().reshape(n_eval, n_eval)
    f2_true = f2_true_t.detach().cpu().numpy().reshape(n_eval, n_eval)
    f1_pred = f1_pred_t.detach().cpu().numpy().reshape(n_eval, n_eval)
    f2_pred = f2_pred_t.detach().cpu().numpy().reshape(n_eval, n_eval)
    div_pred = div_pred_t.detach().cpu().numpy().reshape(n_eval, n_eval)

    err_mag = np.sqrt((f1_pred - f1_true) ** 2 + (f2_pred - f2_true) ** 2)
    div_l2 = np.sqrt(np.mean(div_pred ** 2))
    div_max = np.max(np.abs(div_pred))

    print(f"relative L2 error (vector field) = {rel_l2:.6e}")
    print(f"divergence RMS                = {div_l2:.6e}")
    print(f"divergence max abs            = {div_max:.6e}")


    np.save(
        os.path.join(out_dir, "hpinn_divfree_loss.npy"),
        np.array(loss_history, dtype=np.float64),
    )
    np.savez(
        os.path.join(out_dir, "hpinn_divfree_eval.npz"),
        X1=X1,
        X2=X2,
        f1_true=f1_true,
        f2_true=f2_true,
        f1_pred=f1_pred,
        f2_pred=f2_pred,
        div_pred=div_pred,
        err_mag=err_mag,
    )
    torch.save(model.state_dict(), os.path.join(out_dir, "hpinn_divfree_model.pt"))


    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    ax11, ax12 = axes[0, 0], axes[0, 1]
    ax21, ax22 = axes[1, 0], axes[1, 1]

    mag_true = np.sqrt(f1_true ** 2 + f2_true ** 2)
    mag_pred = np.sqrt(f1_pred ** 2 + f2_pred ** 2)
    mag_p98 = max(
        float(np.percentile(mag_true, 98.0)),
        float(np.percentile(mag_pred, 98.0)),
    )
    mag_norm = Normalize(vmin=0.0, vmax=max(mag_p98, 1e-6))

    q1, _ = pretty_quiver_panel(
        ax11, X1, X2, f1_true, f2_true, lb=lb, rb=rb, title="True vector field", mag_norm=mag_norm
    )
    q2, _ = pretty_quiver_panel(
        ax12, X1, X2, f1_pred, f2_pred, lb=lb, rb=rb, title="hPINN predicted vector field", mag_norm=mag_norm
    )
    cbar_vec = fig.colorbar(q2, ax=[ax11, ax12], fraction=0.03, pad=0.02)
    cbar_vec.set_label("|f| (shared scale)")

    ax21.set_title("|prediction error|")
    pcm1 = ax21.pcolormesh(X1, X2, err_mag, shading="auto", cmap="magma")
    fig.colorbar(pcm1, ax=ax21, fraction=0.046, pad=0.04, label="error magnitude")
    ax21.set_xlabel("x1")
    ax21.set_ylabel("x2")
    ax21.set_xlim(lb, rb)
    ax21.set_ylim(lb, rb)
    ax21.set_aspect("equal", adjustable="box")
    ax21.grid(alpha=0.12, linewidth=0.6)

    ax22.set_title("divergence of hPINN prediction")
    div_lim = np.percentile(np.abs(div_pred), 99.0)
    div_lim = max(float(div_lim), 1e-10)
    pcm2 = ax22.pcolormesh(
        X1, X2, div_pred, shading="auto", cmap="coolwarm", vmin=-div_lim, vmax=div_lim
    )
    fig.colorbar(pcm2, ax=ax22, fraction=0.046, pad=0.04, label="divergence")
    ax22.set_xlabel("x1")
    ax22.set_ylabel("x2")
    ax22.set_xlim(lb, rb)
    ax22.set_ylim(lb, rb)
    ax22.set_aspect("equal", adjustable="box")
    ax22.grid(alpha=0.12, linewidth=0.6)

    fig_path = os.path.join(out_dir, "hpinn_divfree_vector_field.png")
    fig.savefig(fig_path, dpi=180)
    print("saved figure to", fig_path)

    plt.figure(figsize=(7, 4))
    plt.plot(loss_history, lw=1.5)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.title("hPINN training loss")
    plt.grid(alpha=0.3)
    loss_fig = os.path.join(out_dir, "hpinn_divfree_loss.png")
    plt.tight_layout()
    plt.savefig(loss_fig, dpi=180)
    print("saved loss figure to", loss_fig)

    return rel_l2, div_l2, div_max


if __name__ == "__main__":
    device = set_device()
    print("device:", device)
    train_hpinn_divfree(
        a=2.0,
        lb=-1.0,
        rb=1.0,
        n_train=12000,
        batch_size=1024,
        adam_epochs=3000,
        lbfgs_iters=300,
        lr=1e-3,
        n_eval=121,
        out_dir=None,
        device=device,
    )


