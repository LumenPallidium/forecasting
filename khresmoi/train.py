import torch
import einops
from tqdm import tqdm
from data_stream import TimeSeriesGenerator
from hippo_forecaster import HiPPOTransformer

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    day_len = 144
    n_days = 5
    n_series = 100
    n_epochs = 20
    day_dim = 32
    steps_per_epoch = 512
    batch_size = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ts_gen = TimeSeriesGenerator(seed=42, batch_size = batch_size)
    hippo_transformer = HiPPOTransformer(day_dim,
                                         day_len,
                                         n_channels = n_series,
                                         context_time = n_days).to(device)
    
    optimizer = torch.optim.Adam(hippo_transformer.parameters(), lr = 1e-4)
    pbar = tqdm(range(n_epochs * steps_per_epoch))

    for epoch in range(n_epochs):
        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            mixed_series = ts_gen.generate(n_series, day_len * (n_days + 1))
            # b c (t l) -> b c t l
            mixed_series = einops.rearrange(mixed_series,
                                            "b c (t l) -> b c t l",
                                            t = n_days + 1)

            with torch.no_grad():
                x = mixed_series[:, :, :-1, :]
                x_next = mixed_series[:, :, 1:, :]
                x = hippo_transformer.encode(x).to(device)
                x_next = hippo_transformer.encode(x_next).to(device)

            x_next_hat = hippo_transformer(x)
            loss = torch.nn.functional.mse_loss(x_next_hat, x_next)
            loss.backward()
            optimizer.step()
            
            pbar.set_description(f"Epoch {epoch}| Step {step}| Loss {loss.item()}")
            pbar.update(1)
