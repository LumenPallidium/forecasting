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
                                                             in_dim))

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
                                                             in_dim))

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
        assert lookback_steps % num_steps == 0, "Lookback steps must be divisible by num steps"
        self.lookback_steps = lookback_steps

        self.measurement_functions = measurement_functions
        self.n_measurements = len(measurement_functions)
        self.dim = in_dim * self.n_measurements

        self.encoder = KNFEncoder(in_dim, num_steps, measurement_functions)
        self.decoder = KNFDecoder(in_dim, num_steps, self.n_measurements)

        self.koopman_global = torch.nn.Parameter(torch.randn(self.dim,
                                                             self.dim,))
        self.koopman_local = Transformer(self.in_dim,
                                         transformer_layers,
                                         context = self.lookback_steps,
                                         causal = True,
                                         heads = 2,
                                         dropout = 0.1)
        
    def forward(self, x):
        batch_size, total_steps, dim = x.shape
        n_chunks = total_steps // self.num_steps
        x = x.view(batch_size, n_chunks, self.num_steps, dim)
        v = self.encoder(x)

        v_hat = torch.einsum("...j,ji->...i",
                             v.view(batch_size, n_chunks, -1),
                             self.koopman_global)
        v_hat = v_hat.view(batch_size, n_chunks, self.n_measurements, self.in_dim)

        # TODO add the local koopman operator

        x_hat = self.decoder(v_hat)
        x_hat = x_hat.view(batch_size, -1, self.in_dim)
        return x_hat
        
if __name__ == "__main__":
    knf = KNF(3, 2, 4)
    x = torch.randn(8, 10, 3)

    x_hat = knf(x)
    print(x_hat.shape)