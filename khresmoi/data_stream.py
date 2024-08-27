import torch
from tqdm import tqdm
 
class FitzHughNagumoDS:
    """
    The FitzHugh-Nagumo equation is a simplified model of the electrical activity of a neuron.

    Here is used as a simple dynamical system to test the Koopman learning model.
    """
    def __init__(self, a = 0.7, b = 0.8, c = 0.08, I = 0.8):
        self.a = a
        self.b = b
        self.c = c
        self.I = I

    def sample(self, batch_size = 256, T = 100, dt = 0.01, x0 = None):
        with torch.no_grad():
            if x0 is None:
                x0 = torch.randn(batch_size, 2)
            x = x0
            x_t = [x.clone()]
            for t in range(T - 1):
                x[:, 0].add_(dt * (-(x[:, 0] ** 3) / 3 + x[:, 0] - x[:, 1] + self.I))
                x[:, 1].add_(dt * self.c * (x[:, 0] - self.b * x[:, 1] + self.a))

                x_t.append(x.clone())
            x_t = torch.stack(x_t, dim = 1)

        return x_t # batch, T, 2
    

def train_on_ds(model, ds, n_steps = 1000, delay = 10, batch_size = 256):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

    losses = []
    pbar = tqdm(range(n_steps))

    for _ in range(n_steps):
        optimizer.zero_grad()

        sample = ds.sample(T = delay + 1,
                           batch_size = batch_size).to(device)
        x_t = sample[:, :-1, :].detach().clone()
        x_t1 = sample[:, 1:, :].detach().clone().requires_grad_(True)

        loss = model.get_loss(x_t, x_t1)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        pbar.set_description(f"{loss.item():.4f}")
        pbar.update()

    return losses