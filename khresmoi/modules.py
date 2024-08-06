import torch
    
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
                 dropout = 0.,
                 activation = torch.nn.GELU):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, hidden_dim),
            activation(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
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

    def forward(self, x, y = None):
        """Input shape is (batch, seq_len, dim)"""
        x = self.norm(x)

        if self.cross and (not y is None):
            q, k, v = self.W_q(x), self.W_k(y), self.W_v(y)
        else:
            q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
        
        output, _ = self.mha(q, k, v, need_weights=False)

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
                 context = None,
                 activation = torch.nn.GELU,
                 ema_decay = 0.996,
                 first_layer_norm = True,
                 cross = False,):
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.cross = cross

        self.ema_decay = ema_decay

        self.has_util_norm = False

        if first_layer_norm:
            self.norm = torch.nn.LayerNorm(dim)
        else:
            self.norm = torch.nn.Identity()

        if positional_embedding and (context is not None):
            self.pos_embedding = torch.nn.Parameter(torch.randn(1, context, dim))
        else:
            self.register_buffer("pos_embedding", torch.zeros(1, 1, dim))

        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(torch.nn.ModuleList([
                Attention(dim, n_heads = heads, dropout = dropout, cross = cross),
                MLP(dim, dim, dropout = dropout, activation = activation)
            ]))

    def forward(self, x, y = None, stop_at = None, pos_embedding = None):
        """Transformer forward. Can stop at a certain layer for layer-dropout,
        as well as be supplied with a positional embedding (e.g. for shared
        positional embeddings between models)"""
        if pos_embedding is None:
            pos_embedding = self.pos_embedding
        x = self.norm(x) + pos_embedding

        for i, (attention, ff) in enumerate(self.layers):
            x = x + attention(x, y = y)
            x = x + ff(x)

            y = None # disable cross attention after first layer
            if (stop_at is not None) and (i >= (stop_at - 1)):
                break
        return x
    
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
    
    