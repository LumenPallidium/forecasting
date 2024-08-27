import torch
from modules import MLP, Transformer

MEASUREMENT_FUNCTIONS = [torch.sin, torch.cos, torch.exp,
                         torch.sigmoid, torch.tanh]

class KNFEncoder(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 num_steps,
                 measurement_functions = MEASUREMENT_FUNCTIONS,):
        super().__init__()
        self.in_dim = in_dim
        self.num_steps = num_steps
        self.measurement_functions = measurement_functions
        self.n_measurements = len(measurement_functions)

        self.encoder_matrix = torch.nn.Parameter(torch.randn(self.n_measurements,
                                                             num_steps,
                                                             in_dim,
                                                             num_steps,
                                                             in_dim) * 1e-2)

    def forward(self, x):
        """
        Input shape is (batch_size, num_steps, in_dim)
        """
        # (batch_size, num_steps, in_dim) -> (batch_size, n_measurements, num_steps, in_dim)
        x_mod = torch.einsum("...dj,ingdj->...ing", x, self.encoder_matrix)
        # (batch_size, n_measurements, num_steps, in_dim) -> (batch_size, n_measurements, in_dim)
        v = torch.einsum("...nlj,...lj->...nj", x_mod, x)
        v = torch.stack([f(v[..., i, :]) for i, f in enumerate(self.measurement_functions)], dim=-2)
        return v # (batch_size, n_measurements, in_dim)
    
class KNFDecoder(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 num_steps,
                 n_measurements):
        super().__init__()
        self.in_dim = in_dim
        self.num_steps = num_steps
        self.n_measurements = n_measurements

        self.encoder_matrix = torch.nn.Parameter(torch.randn(num_steps,
                                                             in_dim,
                                                             self.n_measurements,
                                                             in_dim) * 1e-2)

    def forward(self, v):
        x = torch.einsum("...nd,ijnd->...ij",
                         v,
                         self.encoder_matrix)
        return x

class KNF(torch.nn.Module):
    """
    The Koopman Neural Forecaster (KNF) model, from the paper:
    https://arxiv.org/pdf/2210.03675

    Parameters:
    ----------
    in_dim: int
        The dimension of the input data
    num_steps: int
        The number of steps at a time that are supplied
    lookback_steps: int
        The number of steps to look back in the past
    
    """
    def __init__(self,
                 in_dim,
                 num_steps,
                 lookback_steps,
                 transformer_layers = 3,
                 measurement_functions = MEASUREMENT_FUNCTIONS,):
        super().__init__()
        self.in_dim = in_dim
        self.num_steps = num_steps
        assert num_steps % lookback_steps == 0, "num_steps must be divisible by lookback_steps"
        self.lookback_steps = lookback_steps

        self.measurement_functions = measurement_functions
        self.n_measurements = len(measurement_functions)
        self.dim = in_dim * self.n_measurements

        self.encoder = KNFEncoder(in_dim, num_steps, measurement_functions)
        self.decoder = KNFDecoder(in_dim, num_steps, self.n_measurements)

        self.koopman_global = torch.nn.Parameter(torch.randn(self.dim,
                                                             self.dim,) * 1e-2)
        self.koopman_local = Transformer(self.in_dim * self.n_measurements,
                                         transformer_layers,
                                         context = self.lookback_steps,
                                         causal = True,
                                         heads = 2,
                                         dropout = 0.1)
        
    def forward(self, x):
        batch_size, total_steps, dim = x.shape
        n_chunks = total_steps // self.num_steps
        #TODO pad?
        x = x.view(batch_size, n_chunks, self.num_steps, dim)
        v = self.encoder(x).view(batch_size, n_chunks, -1)

        v_hat_global = torch.einsum("...j,ji->...i",
                             v,
                             self.koopman_global)
        
        v_local = v.view(batch_size * n_chunks // self.lookback_steps, self.lookback_steps, -1)
        v_hat_local = self.koopman_local(v_local).view(batch_size, n_chunks, -1)

        v_hat = v_hat_global + v_hat_local
        v_hat = v_hat.view(batch_size, n_chunks, self.n_measurements, self.in_dim)

        x_hat = self.decoder(v_hat)
        x_hat = x_hat.view(batch_size, -1, self.in_dim)
        return x_hat
    
    def get_loss(self, x_t, x_t1):
        x_t1_hat = self(x_t)
        return torch.nn.functional.mse_loss(x_t1_hat, x_t1)
        
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from data_stream import FitzHughNagumoDS, train_on_ds
    DELAY = 16
    LOOKBACK_DELAY = 4
    N_STEPS = 1000
    BATCH_SIZE = 512
    HIDDEN_DIM = 32
    HIDDEN_MULTS = [1, 2, 2, 2, 2, 1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = KNF(2, DELAY, LOOKBACK_DELAY)
    ds = FitzHughNagumoDS()

    losses = train_on_ds(model, ds,
                         n_steps = N_STEPS,
                         delay = DELAY,
                         batch_size = BATCH_SIZE)
    
    smooth_losses = np.convolve(losses, np.ones(100) / 100, mode = "valid")
    plt.plot(smooth_losses)