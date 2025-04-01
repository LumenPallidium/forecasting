from dataclasses import dataclass
from typing import Dict
import torch
from collections import defaultdict
    
class ComplexGELU(torch.nn.Module):
    """
    Complex GELU activation function.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        new_real = torch.nn.functional.gelu(x.real)
        # need to do this to avoid in-place operation
        y = torch.complex(new_real, x.imag)
        return y
    
class ComplexReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        new_real = torch.nn.functional.relu(x.real)
        y = torch.complex(new_real, x.imag)
        return y
    
class ComplexLayerNorm(torch.nn.Module):
    def __init__(self,
                 dim,
                 normalize_imag = False,
                 eps = 1e-6):
        super().__init__()
        self.real_layer_norm = torch.nn.LayerNorm(dim, eps = eps)
        # imaginary values may not need to be normalized
        if normalize_imag:
            self.imag_layer_norm = torch.nn.LayerNorm(dim, eps = eps)
        else:
            self.imag_layer_norm = torch.nn.Identity()

    def forward(self, x):
        real = x.real
        imag = x.imag

        real = self.real_layer_norm(real)
        imag = self.imag_layer_norm(imag)

        return torch.complex(real, imag)
    
class FunGen(torch.nn.Module):
    def __init__(self, dim, out_dim = None, bias = True, complex = False):
        super().__init__()
        self.dim = dim
        out_dim = out_dim if out_dim is not None else dim
        self.out_dim = out_dim
        dtype = torch.complex64 if complex else None

        self.linear = torch.nn.Linear(dim, out_dim,
                                      bias = bias,
                                      dtype = dtype)
        self.log_linear = torch.nn.Linear(dim, dim,
                                          bias = bias,
                                          dtype = dtype)
        self.exp_linear = torch.nn.Linear(dim, out_dim,
                                          bias = bias,
                                          dtype = dtype)
        self.out_bias = torch.nn.Parameter(torch.zeros(out_dim,
                                                       dtype = dtype))

        if complex:
            self.relu = ComplexReLU()
        else:
            self.relu = torch.nn.ReLU()

    def forward(self, x):
        y_linear = self.linear(x)
        log_x = torch.log(self.relu(self.log_linear(x)) + 1e-6)

        y_log = self.exp_linear(torch.exp(log_x))
        y = y_linear * y_log + self.out_bias
        return y

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1)

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

class SymLog(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return symlog(x)
    
class SymExp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return symexp(x)

class MLP(torch.nn.Module):
    """An MLP layer.
    
    Parameters
    ----------
    dim : int
        The dimension of the input and output
    hidden_dim : int
        The dimension of the hidden layer
    dropout : float, optional
        The dropout rate, by default 0.
    activation : torch.nn.Module, optional
        The activation function, by default torch.nn.GELU"""
    def __init__(self, 
                 dim, 
                 hidden_dim,
                 out_dim = None,
                 dropout = 0.,
                 activation = torch.nn.GELU,
                 residual = True):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        else:
            residual = False
        self.residual = residual
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, hidden_dim),
            activation(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, out_dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        if self.residual:
            return x + self.net(x)
        return self.net(x)
    
    def ema(self, other, decay = 0.996):
        for p, p_other in zip(self.parameters(), other.parameters()):
            p.data = decay * p.data + (1 - decay) * p_other.data
    
class Attention(torch.nn.Module):
    """Based on ViT implementation from Phil Wang:
    https://github.com/lucidrains/musiclm-pytorch/blob/main/musiclm_pytorch/musiclm_pytorch.py
    
    Parameters
    ----------
    dim : int
        The dimension of the input and output
    dim_head : int, optional
        The dimension of the subspace for each head, by default 64
    n_heads : int, optional
        The number of heads, by default 8
    dropout : float, optional
        The dropout rate, by default 0.
    bias : bool, optional
        Whether to use bias in the linear layers, by default False
    cross : bool, optional
        Whether this module uses cross attention, by default False
    """
    def __init__(self, 
                 dim,
                 n_heads = 8,
                 dropout = 0.,
                 bias = False,
                 cross = False,):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dim_head = dim // n_heads
        self.cross = cross

        self.dropout = dropout
        self.inner_dim = self.dim_head * n_heads

        self.norm = torch.nn.LayerNorm(dim)

        self.W_q = torch.nn.Linear(dim, self.inner_dim, bias = bias)
        self.W_k = torch.nn.Linear(dim, self.inner_dim, bias = bias)
        self.W_v = torch.nn.Linear(dim, self.inner_dim, bias = bias)
        self.W_o = torch.nn.Linear(self.inner_dim, dim, bias = bias)

        self.mha = torch.nn.MultiheadAttention(dim,
                                               n_heads,
                                               dropout = dropout,
                                               batch_first=True)
        

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, y = None, mask = None):
        """Input shape is (batch, seq_len, dim)"""
        x = self.norm(x)

        if self.cross and (not y is None):
            q, k, v = self.W_q(x), self.W_k(y), self.W_v(y)
        else:
            q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
        
        output, _ = self.mha(q, k, v,
                             need_weights=False,
                             attn_mask=mask)

        output = self.W_o(output)

        return self.dropout(output)

    
class Transformer(torch.nn.Module):
    """A residual transformer with attention and feed forward layers.
    
    Parameters
    ----------
    dim : int, optional
        The dimension of the residual stream
    depth : int, optional
        The number of attention and feed forward layers
    heads : int, optional
        The number of attention heads, by default 8
    head_dim : int, optional
        The dimension of the subspaces of the attention heads, by default 64
    dropout : float, optional
        The dropout rate, by default 0.
    positional_embedding : bool, optional
        Whether to use a positional embedding, by default True
    causal : bool, optional
        Whether to use causal attention, by default False
    context : int, optional
        The number of context frames, by default None
    activation : torch.nn.Module, optional
        The activation function, by default torch.nn.GELU
    """
    def __init__(self, 
                 dim = 512, 
                 depth = 4, 
                 heads = 8, 
                 dropout = 0.4,
                 positional_embedding = True,
                 causal = False,
                 context = None,
                 cross_context = None,
                 activation = torch.nn.GELU,
                 ema_decay = 0.996,
                 first_layer_norm = True,
                 cross = False):
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.cross = cross
        self.causal = causal
        self.context = context
        self.cross_context = cross_context

        self.ema_decay = ema_decay

        self.has_util_norm = False

        if first_layer_norm:
            self.norm = torch.nn.LayerNorm(dim)
            self.cross_norm = torch.nn.LayerNorm(dim)
        else:
            self.norm = torch.nn.Identity()
            self.cross_norm = torch.nn.Identity()

        if positional_embedding and (context is not None):
            self.pos_embedding = torch.nn.Parameter(torch.randn(1, context, dim))
        else:
            self.register_buffer("pos_embedding", torch.zeros(1, 1, dim))

        if positional_embedding and (cross_context is not None):
            self.pos_embedding_cross = torch.nn.Parameter(torch.randn(1, cross_context, dim))
        else:
            self.register_buffer("pos_embedding_cross", torch.zeros(1, 1, dim))
        

        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(torch.nn.ModuleList([
                Attention(dim, n_heads = heads, dropout = dropout, cross = cross),
                MLP(dim, dim, dropout = dropout, activation = activation)
            ]))

    def forward(self,
                x,
                y = None,
                stop_at = None,
                pos_embedding = None,
                pos_embedding_cross = None,
                mask = None):
        if pos_embedding is None:
            pos_embedding = self.pos_embedding
        if (y is not None) and (pos_embedding_cross is None):
            y = self.cross_norm(y) + self.pos_embedding_cross
        
        if self.causal and (mask is None):
            mask = torch.triu(torch.ones(x.size(1),
                                         x.size(1)),
                              diagonal = 1).to(x.device,
                                               dtype = torch.bool)

        x = self.norm(x) + pos_embedding

        for i, (attention, ff) in enumerate(self.layers):
            x = x + attention(x, y = y, mask = mask)
            x = x + ff(x)

            if (stop_at is not None) and (i >= (stop_at - 1)):
                break
        return x

@dataclass
class LossComponents:
    """Store and aggregate loss components with minimal overhead."""
    components: Dict[str, torch.Tensor]
    
    def __init__(self):
        self.components = {}
        
    def add(self, name: str, loss: torch.Tensor):
        """Add a loss component."""
        self.components[name] = loss
        
    @property
    def total(self) -> torch.Tensor:
        """Compute total loss for backpropagation."""
        return sum(self.components.values())
    
    def detach_dict(self) -> Dict[str, float]:
        """Get detached float values for logging."""
        return {name: loss.detach().item() 
                for name, loss in self.components.items()}

class LossTracker:
    """Track loss components across epochs/iterations."""
    def __init__(self):
        self.history = defaultdict(list)
        
    def update(self, loss_components: LossComponents):
        """Store the current loss values."""
        detached = loss_components.detach_dict()
        for name, value in detached.items():
            self.history[name].append(value)
        self.history['total'].append(sum(detached.values()))
        
    def get_means(self) -> Dict[str, float]:
        """Get mean values for the current epoch."""
        return {name: sum(values) / len(values) 
                for name, values in self.history.items()}
    
    def reset(self):
        """Reset history for new epoch."""
        self.history.clear()
    
if __name__ == "__main__":
    # Test the transformer
    transformer = Transformer()
    x = torch.randn(10, 100, 512)
    y = torch.randn(10, 100, 512)
    out = transformer(x, y)
    print(out.shape)
    
    # Test the MLP
    mlp = MLP(512, 512)
    out = mlp(x)
    print(out.shape)
    
    # Test the attention
    attention = Attention(512)
    out = attention(x)
    print(out.shape)
    
    