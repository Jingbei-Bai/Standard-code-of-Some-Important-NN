import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(0)
np.random.seed(0)
T=5.0
N=501
t=np.linspace(0.0,T,N)
dt=t[1]-t[0]
y_true=np.exp(-t)
dy_true=-y_true
device=torch.device("cuda")
X=torch.tensor(y_true[:-1].reshape(-1,1),dtype=torch.float32,device=device)
Y=torch.tensor(dy_true[:-1].reshape(-1,1),dtype=torch.float32,device=device)
class MultistepNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(1,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,x):
        return self.net(x)
model=MultistepNN().to(device)
opt=optim.Adam(model.parameters(),lr=1e-3)
loss_fn=nn.MSELoss()
for epoch in range(2000):
    pred=model(X)
    loss=loss_fn(pred,Y)
    opt.zero_grad()
    loss.backward()
    opt.step()
methods=['AB','AM','BDF']
for METHOD in methods:
    with torch.no_grad():
        if METHOD=='AB':
            ys=[torch.tensor(y_true[0],dtype=torch.float32,device=device),torch.tensor(y_true[1],dtype=torch.float32,device=device)]
            f=[model(ys[0].unsqueeze(0)),model(ys[1].unsqueeze(0))]
            for n in range(1,N-1):
                if n<2:
                    k1=model(ys[-1].unsqueeze(0))
                    k2=model((ys[-1]+0.5*dt*k1).unsqueeze(0))
                    k3=model((ys[-1]+0.5*dt*k2).unsqueeze(0))
                    k4=model((ys[-1]+dt*k3).unsqueeze(0))
                    y_next=ys[-1]+(dt/6.0)*(k1+2*k2+2*k3+k4)
                else:
                    f2=f[-1]
                    f1=f[-2]
                    f0=f[-3]
                    y_next=ys[-1]+dt*(23.0/12.0*f2-16.0/12.0*f1+5.0/12.0*f0)
                ys.append(y_next.squeeze())
                f.append(model(ys[-1].unsqueeze(0)))
            pred=np.array([v.item() for v in ys[:N]])
        elif METHOD=='AM':
            ys=[torch.tensor(y_true[0],dtype=torch.float32,device=device),torch.tensor(y_true[1],dtype=torch.float32,device=device)]
            f=[model(ys[0].unsqueeze(0)),model(ys[1].unsqueeze(0))]
            for n in range(1,N-1):
                if n<2:
                    k1=model(ys[-1].unsqueeze(0))
                    k2=model((ys[-1]+0.5*dt*k1).unsqueeze(0))
                    k3=model((ys[-1]+0.5*dt*k2).unsqueeze(0))
                    k4=model((ys[-1]+dt*k3).unsqueeze(0))
                    y_pred=ys[-1]+(dt/6.0)*(k1+2*k2+2*k3+k4)
                else:
                    f2=f[-1]
                    f1=f[-2]
                    f0=f[-3]
                    y_pred=ys[-1]+dt*(23.0/12.0*f2-16.0/12.0*f1+5.0/12.0*f0)
                f_pred=model(y_pred.unsqueeze(0))
                y_next=ys[-1]+dt*(5.0/12.0*f_pred+8.0/12.0*f[-1]-1.0/12.0*f[-2])
                ys.append(y_next.squeeze())
                f.append(model(ys[-1].unsqueeze(0)))
            pred=np.array([v.item() for v in ys[:N]])
        else:
            ys=[torch.tensor(y_true[0],dtype=torch.float32,device=device),torch.tensor(y_true[1],dtype=torch.float32,device=device)]
            f=[model(ys[0].unsqueeze(0)),model(ys[1].unsqueeze(0))]
            for n in range(1,N-1):
                if n<1:
                    k1=model(ys[-1].unsqueeze(0))
                    k2=model((ys[-1]+0.5*dt*k1).unsqueeze(0))
                    k3=model((ys[-1]+0.5*dt*k2).unsqueeze(0))
                    k4=model((ys[-1]+dt*k3).unsqueeze(0))
                    y_next=ys[-1]+(dt/6.0)*(k1+2*k2+2*k3+k4)
                else:
                    f_n=f[-1]
                    f_nm1=f[-2]
                    y_pred=ys[-1]+dt*(3.0/2.0*f_n-1.0/2.0*f_nm1)
                    yk=y_pred
                    for it in range(10):
                        fy=model(yk.unsqueeze(0))
                        yk=(2.0/3.0)*(2.0*ys[-1]-0.5*ys[-2]+dt*fy)
                    y_next=yk
                ys.append(y_next.squeeze())
                f.append(model(ys[-1].unsqueeze(0)))
            pred=np.array([v.item() for v in ys[:N]])
    mse=np.mean((pred-y_true)**2)
    print(f"METHOD={METHOD} MSE={mse:.6e}")

