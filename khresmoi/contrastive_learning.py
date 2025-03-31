import torch
from torch.nn.utils.parametrizations import orthogonal
from torch.nn.functional import normalize, cross_entropy

def matrix_power(diagonal, rotation, n_tensor):
    # n_tensor indexes batch
    power_diag = torch.diag(diagonal)[None, :, :].pow(n_tensor[:, None, None])
    power_A = torch.einsum("ij,bjk,lk->bik",
                           rotation,
                           power_diag,
                           rotation)
    return power_A

class PowerLinear(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.diag = torch.nn.Parameter(torch.ones(dim,
                                                  dtype=torch.float32))
        # orthogonal parameterization requires a module, so using a linear but only the weight is used
        rot = torch.nn.Linear(dim, dim, bias = False)
        rot = orthogonal(rot)
        self.rot = rot

    def forward(self, x, n):
        rot = self.rot.weight
        power_A = matrix_power(self.diag, rot, n)
        return torch.einsum("bij,bj->bi",
                            power_A,
                            x)
    
class FixedTimeEmb(torch.nn.Module):
    def __init__(self, dim, max_len = 128):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.A = torch.nn.Linear(2 * dim, dim)
        self.emb = torch.nn.Embedding(max_len, dim)

    def forward(self, x, n):
        n_emb = self.emb(n)
        x = torch.cat([x, n_emb], dim = -1)
        return self.A(x)

def sample_geometric(decay, n = 1, max_len = 100):
    dist = torch.distributions.Geometric(probs = 1 - decay)
    sample = dist.sample((n,)) + 1
    sample = torch.clamp(sample, 0, max_len)
    if n == 1:
        sample = sample.item()
    return sample

def infonce_loss(x, x_plus, temperature=0.5, l2_decay=1e-6):
    x_norm = normalize(x, dim=-1)
    x_plus_norm = normalize(x_plus, dim=-1)
    
    sim_matrix = torch.matmul(x_norm, x_plus_norm.T)
    logits = sim_matrix / temperature
    labels = torch.arange(x.size(0)).to(x.device)
    
    loss = cross_entropy(logits, labels)
    
    if l2_decay > 0:
        l2_loss = 0.5 * ((x_norm ** 2).sum(-1).mean() + (x_plus_norm ** 2).sum(-1).mean())
        loss += l2_decay * l2_loss
        
    return loss

def infonce_no_resub(x, x_plus, l2_decay = 1e-6):
    # batch, dim
    positive_term = (x - x_plus).pow(2).sum(-1)
    negative_term = (x[:, None, :] - x_plus[None, :, :]).pow(2).mean(-1)
    negative_term.fill_diagonal_(0)
    negative_term = 0.5 * (torch.logsumexp(-negative_term, dim=1) + torch.logsumexp(-negative_term.T, dim=1))
    if l2_decay > 0:
        l2_loss = 0.5 * (x_plus ** 2).sum(-1).mean()
        positive_term += l2_decay * l2_loss
    return (positive_term + negative_term).mean()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from data_stream import FitzHughNagumoDS, TimeSpiralDS
    from modules import MLP
    from tqdm import tqdm
    n_epochs = 20
    steps_per_epoch = 500
    batch_size = 1024
    max_t = 128
    dim = 2
    embed_dim = 8
    dims = [16, 16, 16]
    dt = 0.003
    gamma = 0.3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = TimeSpiralDS()

    net = [torch.nn.Linear(dim, embed_dim)]
    net = net + [MLP(embed_dim, dims[i],
                     activation = torch.nn.LeakyReLU) for i in range(len(dims))]
    net_plus = PowerLinear(embed_dim).to(device) 
    # net_plus = [MLP(dim, dims[i],
    #                 activation = torch.nn.LeakyReLU) for i in range(len(dims))]
    net = torch.nn.Sequential(*net).to(device)
    # net_plus = torch.nn.Sequential(*net_plus).to(device)

    optimizer = torch.optim.Adam(list(net.parameters()) + list(net_plus.parameters()),
                                 weight_decay=1e-5,
                                 lr = 1e-4)
    losses = []
    pbar = tqdm(range(n_epochs * steps_per_epoch))
    for epoch in range(n_epochs):
        for step in range(steps_per_epoch):
            traj = ds.sample(batch_size = batch_size,
                             dt = dt,
                             T = max_t)
            traj = traj.to(device)
            traj_len = sample_geometric(gamma,
                                        n = batch_size,
                                        max_len = max_t)
            traj_len = traj_len.to(torch.long).to(device)
                                        
            x = traj[:, 0, :]
            x_plus = traj[torch.arange(batch_size), traj_len, :]

            optimizer.zero_grad()
            # x_emb = net(x)
            # x_emb_plus = net_plus(x_plus)
            # loss = infonce_no_resub(x_emb, x_emb_plus)

            with torch.no_grad():
                x_plus_emb = net(x_plus)
            loss = infonce_no_resub(net_plus(net(x), traj_len),
                                    x_plus_emb)
                                    
            loss.backward()
            # clip gradient
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.2)
            torch.nn.utils.clip_grad_norm_(net_plus.parameters(), 0.2)

            optimizer.step()
            losses.append(loss.item())
            pbar.update(1)
            pbar.set_description(f"Epoch {epoch} Loss {loss.item():.2f}")

    pbar.close()

    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    ax[0].plot(losses)
    ax[0].set_xlabel("Step")
    ax[0].set_ylabel("Loss")

    # choose random point from the trajectory
    idx = torch.randint(0, traj.shape[0], (1,))
    x_ex = traj[idx, 5, :]
    x_ex = x_ex.to(device)
    x_emb = net(x_ex)

    x_min = traj[idx, :, 0].min()
    x_min *= (1 - torch.sign(x_min) * 0.6)
    x_max = traj[idx, :, 0].max()
    x_max *= (1 + torch.sign(x_max) * 0.6)
    y_min = traj[idx, :, 1].min()
    y_min *= (1 - torch.sign(y_min) * 0.6)
    y_max = traj[idx, :, 1].max()
    y_max *= (1 + torch.sign(y_max) * 0.6)

    x_space = torch.linspace(x_min, x_max,
                             20)
    y_space = torch.linspace(y_min, y_max,
                             20)
    X, Y = torch.meshgrid(x_space, y_space)
    XY = torch.stack([X, Y], dim = -1).to(device)
    
    x_emb_plus = net_plus(x_emb, torch.tensor([1],
                                              device = device))
    XY_emb = net(XY.view(-1, 2))
    XY_align = (XY_emb - x_emb).pow(2).sum(-1)
    XY_align = XY_align.softmax(dim = 0).view(20, 20)
    XY_align = XY_align.cpu().detach().numpy()
    # XY_emb = net_plus(XY.view(-1, 2))
    # diff = (XY_emb - x_emb).pow(2).sum(-1)
    # XY_align = diff.view(20, 20)
    # XY_align = XY_align.cpu().detach().numpy()


    ax[1].contourf(X.cpu().detach().numpy(),
                   Y.cpu().detach().numpy(),
                   XY_align)
    ax[1].plot(traj[idx, :, 0].cpu().detach().numpy()[0],
               traj[idx, :, 1].cpu().detach().numpy()[0],
               color = "red")
    ax[1].scatter(traj[idx, 5, 0].cpu().detach().numpy(),
                  traj[idx, 5, 1].cpu().detach().numpy(),
                  color = "red",
                  marker = "o")

    ax[1].scatter(traj[idx, -1, 0].cpu().detach().numpy(),
                  traj[idx, -1, 1].cpu().detach().numpy(),
                  color = "red",
                  marker = "*")

