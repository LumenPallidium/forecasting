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
        x_mod = torch.einsum("bdj,ingdj->bing", x, self.encoder_matrix)
        # (batch_size, n_measurements, num_steps, in_dim) -> (batch_size, n_measurements, in_dim)
        v = torch.einsum("bnlj,blj->bnj", x_mod, x)
        v = torch.stack([f(v[:, i, :]) for i, f in enumerate(self.measurement_functions)], dim=1)
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
        x = torch.einsum("bnd,ijnd->bij", v, self.encoder_matrix)
        return x


class KNF(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 num_steps,
                 lookback_steps,
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
        
    def forward(self, x):
        v = self.encoder(x)
        x = self.decoder(v)
        return x
        
if __name__ == "__main__":
    knf = KNF(3, 2, 4)
    x = torch.randn(8, 2, 3)

    x_hat = knf(x)
    print(x_hat.shape)