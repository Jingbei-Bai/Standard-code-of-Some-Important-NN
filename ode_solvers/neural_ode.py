import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
T=5.0
N=501
t=np.linspace(0.0,T,N)
dt=t[1]-t[0]
y_true=np.exp(-t)
dy_true=-y_true
device=torch.device("cuda")
X=torch.tensor(y_true[:-1].reshape(-1,1),dtype=torch.float32,device=device)
Y=torch.tensor(dy_true[:-1].reshape(-1,1),dtype=torch.float32,device=device)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(1,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
    def forward(self,x):
        return self.net(x)
model=Net().to(device)
opt=optim.Adam(model.parameters(),lr=5e-5)
loss_fn=nn.MSELoss()
for epoch in range(12000):
    pred=model(X)
    loss=loss_fn(pred,Y)
    opt.zero_grad()
    loss.backward()
    opt.step()
with torch.no_grad():
    ys=[torch.tensor([1.0],dtype=torch.float32,device=device)]
    for i in range(N-1):
        y=ys[-1]
        k1=model(y)
        k2=model(y+0.5*dt*k1)
        k3=model(y+0.5*dt*k2)
        k4=model(y+dt*k3)
        y_next=y+(dt/6.0)*(k1+2*k2+2*k3+k4)
        ys.append(y_next)
    pred=np.array([v.item() for v in ys])
print(f"MSE: {np.mean((pred - y_true)**2):.12f}")

