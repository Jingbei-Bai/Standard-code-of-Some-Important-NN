import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def soliton(x,t,c=1.0,x0=0.0):
    k = np.sqrt(c)/2.0
    return 0.5*c/ (np.cosh(k*(x - c*t - x0))**2)

def make_dataset(nx=64, nt=200, T=5.0, n_samples=200, c=1.0):
    x = np.linspace(-20,20,nx)
    dt = T/nt
    seqs = []
    times = np.linspace(0,T,nt+1)
    for s in range(n_samples):
        x0 = np.random.uniform(-5,5)
        traj = []
        for t in times:
            traj.append(soliton(x,t,c,x0))
        seqs.append(np.stack(traj))
    data = np.stack(seqs)

    return data, x, times

class Seq2Seq(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.enc = nn.GRU(input_size,hidden_size,batch_first=True)
        self.dec = nn.GRU(input_size,hidden_size,batch_first=True)
        self.out = nn.Linear(hidden_size,input_size)
    def forward(self,src, tgt_len, teacher_forcing_ratio=0.5):
        batch = src.size(0)
        _,h = self.enc(src)
        decoder_input = src[:,-1,:].unsqueeze(1)
        outputs = []
        hidden = h
        for t in range(tgt_len):
            out,hidden = self.dec(decoder_input,hidden)
            frame = self.out(out.squeeze(1)).unsqueeze(1)
            outputs.append(frame)
            if self.training and np.random.rand() < teacher_forcing_ratio:
                decoder_input = frame
            else:
                decoder_input = frame
        return torch.cat(outputs,dim=1)

def train_model(device):
    data,x,times = make_dataset(nx=64,nt=200,T=5.0,n_samples=400,c=1.0)
    train = data[:320]
    val = data[320:360]
    test = data[360:]
    input_len = 10
    pred_len = 20
    model = Seq2Seq(input_size=64,hidden_size=256).to(device)
    opt = optim.Adam(model.parameters(),lr=1e-3)
    loss_fn = nn.MSELoss()
    epochs = 200
    for e in range(epochs):
        model.train()
        idx = np.random.permutation(train.shape[0])
        batch_size = 16
        total_loss = 0.0
        for i in range(0,train.shape[0],batch_size):
            batch_idx = idx[i:i+batch_size]
            seq = torch.tensor(train[batch_idx,:input_len,:],dtype=torch.float32,device=device)
            tgt = torch.tensor(train[batch_idx,input_len:input_len+pred_len,:],dtype=torch.float32,device=device)
            opt.zero_grad()
            out = model(seq,pred_len,teacher_forcing_ratio=0.5)
            loss = loss_fn(out,tgt)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (e+1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                vseq = torch.tensor(val[:16,:input_len,:],dtype=torch.float32,device=device)
                vout = model(vseq,pred_len,teacher_forcing_ratio=0.0)
                vtrue = torch.tensor(val[:16,input_len:input_len+pred_len,:],dtype=torch.float32,device=device)
                vloss = loss_fn(vout,vtrue).item()
            print(f'epoch {e+1}/{epochs} train_loss={(total_loss/ (train.shape[0]/batch_size)):.6f} val_loss={vloss:.6f}')
    return model, x, times, test, input_len, pred_len

def plot_example(model,x,times,test,input_len,pred_len,device):
    model.eval()
    idx = 0
    seq = torch.tensor(test[idx:idx+1,:input_len,:],dtype=torch.float32,device=device)
    with torch.no_grad():
        pred = model(seq,pred_len,teacher_forcing_ratio=0.0).cpu().numpy()[0]
    true = test[idx,input_len:input_len+pred_len,:]
    t0 = times[input_len]
    plt.figure(figsize=(8,6))
    nt = pred_len
    for k in range(min(6,nt)):
        plt.subplot(3,2,k+1)
        plt.plot(x,true[k],label='true')
        plt.plot(x,pred[k],label='pred')
        plt.title(f't={t0 + k*(times[1]-times[0]):.3f}')
        if k==0:
            plt.legend()
    os.makedirs('ode_solvers_outputs',exist_ok=True)
    out = os.path.join('ode_solvers_outputs','seq2seq_kdv.png')
    plt.tight_layout()
    plt.savefig(out)
    print('saved',out)

if __name__=='__main__':
    device = torch.device("cuda")
    model,x,times,test,input_len,pred_len = train_model(device)
    plot_example(model,x,times,test,input_len,pred_len,device)


