import torch
from modules import MLP, Transformer

MEASUREMENT_FUNCTIONS = [torch.sin, torch.cos, torch.exp,
                         torch.logsumexp, torch.sigmoid, torch.tanh]

class KNF(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 dim,
                 num_steps,
                 lookback_steps,
                 measurement_functions = MEASUREMENT_FUNCTIONS,):
        super().__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.num_steps = num_steps
        assert lookback_steps // num_steps == 0, "Lookback steps must be divisible by num steps"
        self.lookback_steps = lookback_steps

        self.measurement_functions = measurement_functions
        self.n_measurements = len(measurement_functions)


        #TODO
        self.encoder = None
        self.decoder = None

        self.koopman_global = torch.nn.Parameter(torch.randn(dim * in_dim,
                                                             dim * in_dim))