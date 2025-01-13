import torch
import torch.nn.functional as F
import numpy as np
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

    def sample(self, batch_size = 256, T = 100, dt = 0.01, x0 = None, I = None):
        if I is None:
            I = self.I
        with torch.no_grad():
            if x0 is None:
                x0 = torch.randn(batch_size, 2)
            x = x0
            x_t = [x.clone()]
            for t in range(T - 1):
                x[:, 0].add_(dt * (-(x[:, 0] ** 3) / 3 + x[:, 0] - x[:, 1] + I))
                x[:, 1].add_(dt * self.c * (x[:, 0] - self.b * x[:, 1] + self.a))

                x_t.append(x.clone())
            x_t = torch.stack(x_t, dim = 1)

        return x_t # batch, T, 2
    
class CoupledFitzHughNagumoDS:
    def __init__(self, n_copies = 5, a = 0.7, b = 0.8, c = 0.08, I = 0.8):
        self.n_copies = n_copies
        self.a = a
        self.b = b
        self.c = c
        self.I = I

    def sample(self, batch_size = 256, T = 100, dt = 0.01, x0 = None):
        I = self.I
        with torch.no_grad():
            if x0 is None:
                x0 = torch.randn(batch_size, self.n_copies, 2)
            x = x0
            x_t = [x.clone()]
            for t in range(T - 1):
                x[:, :, 0].add_(dt * (-(x[:, :, 0] ** 3) / 3 + x[:, :, 0] - x[:, :, 1] + I))
                x[:, :, 1].add_(dt * self.c * (x[:, :, 0] - self.b * x[:, :, 1] + self.a))

                # they feel the local field
                I = torch.mean(x[:, :, 0])
                x_t.append(x.clone())
            x_t = torch.stack(x_t, dim = 1).view(batch_size, T, -1)

        return x_t # batch, T, 2 * n_copies

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

class TimeSeriesGenerator:
    """
    A simple, nonstationary time series generator, mostly from Claude.
    """
    def __init__(self,
                 scale_max = 1e6,
                 scale_min = 0,
                 var_max = 1e10,
                 max_growth = 10000,
                 temperature = 0.1,
                 min_freq = 0.5,
                 max_freq = 144,
                 max_sin_comps = 30,
                 seed=None):
        self.scale_max = scale_max
        self.scale_min = scale_min
        if var_max < 1:
            var_max = scale_max * var_max
        self.var_max = var_max

        self.min_freq = min_freq
        self.max_freq = max_freq
        self.max_sin_comps = max_sin_comps
        self.max_growth = max_growth
        self.temperature = temperature
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
    def generate_smooth_variance(self,
                                 length,
                                 control_points_per = 20):
        num_points = length // control_points_per + 2
        control_points = torch.rand(num_points) * self.var_max
        
        variance = F.interpolate(
                                control_points.view(1, 1, -1),
                                size=length,
                                mode="linear",
                                align_corners=True
                            ).squeeze()
        
        return variance
        
    def generate_trend(self,
                       length,
                       trend_type="mixed"):
        t = torch.linspace(0, 1, length)

        base = torch.rand(1) * (self.scale_max - self.scale_min) + self.scale_min
        
        if trend_type == "sin":
            n_freqs = torch.randint(1, self.max_sin_comps, (1,)).item()
            freq = torch.rand(n_freqs) * (self.max_freq - self.min_freq) + self.min_freq
            phase = torch.rand(n_freqs) * np.pi
            scale = (torch.rand(n_freqs) / (5 * n_freqs * freq)) * (self.scale_max - self.scale_min) + self.scale_min
            aggs = scale[None, :] * torch.sin(2 * np.pi * freq[None, :] * t[:, None] + phase[None, :])
            return aggs.sum(dim=1) + base
        elif trend_type == "linear":
            slope = torch.rand(1)**2 * 2 - 1
            return self.max_growth * slope * t + base
        elif trend_type == "logistic":
            k = torch.rand(1) * (self.scale_max - self.scale_min) + self.scale_min
            x0 = torch.rand(1) * 0.5 + 0.25
            return 1 / (1 + torch.exp(-k * (t - x0))) + base
        elif trend_type == "normal":
            center = torch.rand(1)
            width = torch.rand(1) * 0.1
            scale = torch.rand(1) * (self.scale_max - self.scale_min) + self.scale_min
            return scale * torch.exp(-0.5 * ((t - center) / width) ** 2) + base
        elif trend_type == "mixed":
            trends = [
                self.generate_trend(length, "sin"),
                self.generate_trend(length, "linear"),
                self.generate_trend(length, "logistic"),
                self.generate_trend(length, "normal")
            ]
            weights = F.softmax(torch.rand(len(trends)) / self.temperature,
                                dim=0)
            return sum(w * trend for w, trend in zip(weights, trends)) + base
            
    def generate_pure_series(self, n, length):
        trends = torch.stack([
            self.generate_trend(length, "mixed") 
            for _ in range(n)
        ])
        
        variances = torch.stack([
            self.generate_smooth_variance(length)
            for _ in range(n)
        ])
        
        noise = torch.randn(n, length) * torch.sqrt(variances)
        noise = torch.cumsum(noise, dim=1)
        
        series = trends + noise
        series[series < self.scale_min] = self.scale_min
        
        return series, variances
        
    def generate_time_varying_covariance(self, n, length, sparsity = 0.5, max_corr = 0.2):
        rotation_speeds = torch.rand(n, n) * (torch.rand(n, n) > sparsity)
        base_correlations = torch.rand(n, n) * max_corr
        base_correlations = (base_correlations + base_correlations.T) / 2
        base_correlations.fill_diagonal_(1.0)
        
        t = torch.linspace(0, 1, length)
        correlations = []
        
        for ti in t:
            rotation = torch.sin(2 * np.pi * rotation_speeds * ti)
            rotation = (rotation + rotation.T) / 2
            rotation.fill_diagonal_(1.0)
            
            corr = base_correlations * (0.7 + 0.3 * rotation)
            corr = (corr + corr.T) / 2
            correlations.append(corr)
            
        return torch.stack(correlations)

    def generate(self, n, length):
        pure_series, variances = self.generate_pure_series(n, length)
        correlations = self.generate_time_varying_covariance(n, length)
        
        mixed_series = []
        for t in range(length):
            corr = correlations[t]
            
            L = torch.linalg.cholesky(corr)
            mixed_t = torch.matmul(L, pure_series[:, t])
            mixed_series.append(mixed_t)
            
        mixed_series = torch.stack(mixed_series, dim=1)
        
        return mixed_series, pure_series, correlations
    
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    day_len = 144
    n_days = 3
    n_series = 100
    ts_gen = TimeSeriesGenerator(seed=42)
    mixed_series, pure_series, correlations = ts_gen.generate(n_series,
                                                              day_len * n_days)

    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    for i in range(100):
        j = i // 10
        k = i % 10
        axes[j, k].plot(mixed_series[i].numpy())
        axes[j, k].set_title(f"Series {i + 1}")
