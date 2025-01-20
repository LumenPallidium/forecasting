import torch

def sample_geometric(decay, n = 1, max_len = 100):
    dist = torch.distributions.Geometric(probs = 1 - decay)
    sample = dist.sample((n,))
    sample = torch.clamp(sample, 0, max_len)
    if n == 1:
        sample = sample.item()
    return sample

def infonce_no_resub(x, x_plus):
    # batch, dim
    positive_term = (x - x_plus).pow(2).sum(-1)
    negative_term = (x[:, None, :] - x_plus[None, :, :]).pow(2).sum(-1)
    negative_term.fill_diagonal_(float("inf"))
    negative_term = 0.5 * (torch.logsumexp(-negative_term, dim=1) + torch.logsumexp(-negative_term.T, dim=1))
    return (positive_term + negative_term).mean()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from data_stream import FitzHughNagumoDS
    from modules import MLP
    from tqdm import tqdm
    n_epochs = 5
    steps_per_epoch = 500
    batch_size = 512
    max_t = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = FitzHughNagumoDS()

    net = MLP(2, 32, out_dim = 16)
    net_plus = torch.nn.Linear(16, 16, bias = False)
    inital_A = net_plus.weight.data.detach().numpy()

    net.to(device)
    net_plus.to(device)
    optimizer = torch.optim.Adam(list(net.parameters()) + list(net_plus.parameters()),
                                 lr = 1e-4)
    losses = []
    pbar = tqdm(range(n_epochs * steps_per_epoch))
    for epoch in range(n_epochs):
        for step in range(steps_per_epoch):
            traj = ds.sample(batch_size = batch_size,
                             T = max_t)
            traj = traj.to(device)
            traj_len = sample_geometric(0.1,
                                        n = batch_size,
                                        max_len = max_t)
            traj_len = traj_len.to(torch.long).to(device)
                                        
            x = traj[:, 0, :]
            x_plus = traj[torch.arange(batch_size), traj_len, :]

            optimizer.zero_grad()
            loss = infonce_no_resub(net(x), net_plus(net(x_plus)))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.update(1)
            pbar.set_description(f"Epoch {epoch} Loss {loss.item():.2f}")

    final_A = net_plus.weight.data.cpu().detach().numpy()
    plt.plot(losses)
