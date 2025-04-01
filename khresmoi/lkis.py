import torch
from torch.nn.functional import mse_loss
from modules import ComplexGELU, ComplexReLU, ComplexLayerNorm, FunGen, Transformer

class DeepMLP(torch.nn.Module):
    """
    Multi-layer perceptron (MLP) model.
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_mults = 3,
                 activation = ComplexGELU(),
                 complex = True):
        """
        Parameters
        ----------
        input_dim : int
            The dimension of the input data.
        output_dim : int
            The dimension of the output data.
        hidden_mults : int or list of int, optional
            The multiplier for the hidden layers. If int, the same multiplier is used for all hidden layers.
            If list, the multipliers are used in the order given. Defaults to 3.
        activation : torch.nn.Module, optional
            The activation function to use. Defaults to GELU.
        complex : bool, optional
            Whether to use complex numbers. Defaults to True.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if isinstance(hidden_mults, int):
            hidden_mults = [hidden_mults]

        dtype = torch.complex64 if complex else None
        layernorm = ComplexLayerNorm if complex else torch.nn.LayerNorm

        layers = []
        in_dim = input_dim
        for mult in hidden_mults:
            layers.append(layernorm(in_dim))
            layers.append(torch.nn.Linear(in_dim, int(in_dim * mult),
                                          dtype = dtype))
            layers.append(activation)
            in_dim = int(in_dim * mult)
        layers.append(torch.nn.Linear(in_dim, output_dim,
                                      dtype = dtype),)
        self.layers = torch.nn.Sequential(*layers)
        

    def forward(self, x):
        return self.layers(x)
     
class CrossAttentionEmbedder(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 in_context,
                 out_context = 2,
                 num_layers = 1,
                 num_heads = 4):
            super().__init__()
            self.in_dim = in_dim
            self.hidden_dim = hidden_dim
            self.num_heads = num_heads
            self.context = out_context

            self.embed = torch.nn.Linear(in_dim, hidden_dim)
            self.transformer = Transformer(hidden_dim,
                                           heads = num_heads,
                                           depth = num_layers,
                                           context = out_context,
                                           cross_context = in_context,
                                           dropout = 0.1,
                                           cross = True,)
            self.mask_token = torch.nn.Parameter(torch.randn(1, 1, hidden_dim))
            
    def forward(self, x):
        x_embed = self.embed(x)
        y_embed = self.mask_token.expand(x_embed.shape[0],
                                         self.context,
                                         -1).clone()
        y_embed = y_embed.to(x.device)
        y_embed = self.transformer(y_embed, y = x_embed)
        return y_embed.transpose(1, 2)


class LKIS(torch.nn.Module):
    """
    The learning Koopman invariant subspaces (LKIS) model.
    Uses deep learning to learn the the measurement operators in Koopman theory.
    From the paper:
    https://arxiv.org/pdf/1710.04340.pdf
    """
    def __init__(self,
                 input_dim,
                 delay,
                 hidden_dim,
                 bottleneck_dim = None,
                 hidden_mults = 3,
                 neural_koopman = True,
                 attention_embed = False,
                 use_decoder = True,
                 alpha = 1,
                 complex =True):
        """
        Parameters
        ----------
        input_dim : int
            The dimension of the input data.
        delay : int
            The delay in the input data i.e. number of time samples
        hidden_dim : int
            The dimension of the hidden layer after timeseries embedding.
        bottleneck_dim : int, optional
            The dimension of the bottleneck layer. If not given, defaults to hidden_dim.
        hidden_mults : int or list of int, optional
            The multiplier for the hidden layers. If int, the same multiplier is used for all hidden layers.
            If list, the multipliers are used in the order given.
        neural_koopman : bool
            Whether or not to use a neural Koopman operator. If false, use delay and pinverse.
        attention_embed : bool, optional
            Whether to use a transformer for embedding or a linear embedding.
        use_decoder : bool, optional
            Whether to use a decoder to reconstruct the input data. Defaults to True.
        alpha : float, optional
            The weight of the reconstruction loss. Defaults to 1.
        complex : bool, optional
            Whether to use complex numbers. Defaults to True.
        """
        super().__init__()
        self.input_dim = input_dim
        self.delay = delay
        self.hidden_dim = hidden_dim
        self.use_decoder = use_decoder
        if bottleneck_dim is None:
            bottleneck_dim = hidden_dim
        self.attention_embed = attention_embed
        self.neural_koopman = neural_koopman
        self.bottleneck_dim = bottleneck_dim
        self.alpha = alpha
        self.complex = complex
        if complex:
            activation = ComplexGELU()
            embed_mult = 2
        else:
            activation = torch.nn.GELU()
            embed_mult = 1

        assert bottleneck_dim % delay == 0, "hidden_dim must be divisible by delay"

        if isinstance(hidden_mults, int):
            hidden_mults = [hidden_mults]
        self.hidden_mults = hidden_mults

        if len(hidden_mults) != 1:
            fun_gen_dim = hidden_mults[-1] * hidden_dim
            hidden_mults = hidden_mults[:-1]
            fun_gen = FunGen(fun_gen_dim, complex = complex)
            self.encoder = torch.nn.Sequential(DeepMLP(hidden_dim, bottleneck_dim,
                                                       hidden_mults, activation = activation,
                                                       complex = complex),
                                               fun_gen)

        else:
            self.encoder = DeepMLP(hidden_dim, bottleneck_dim,
                                   hidden_mults, activation = activation,
                                   complex = complex)

        if attention_embed:
            self.embedder = CrossAttentionEmbedder(input_dim,
                                                   hidden_dim,
                                                   in_context = delay,
                                                   out_context = embed_mult,
                                                   num_layers = 1,
                                                   num_heads = 4)
        else:
            self.embedder = torch.nn.Linear(input_dim * delay, hidden_dim * embed_mult)

        if self.use_decoder:
            hidden_mults = hidden_mults[::-1]
            self.decoder = DeepMLP(bottleneck_dim, hidden_dim,
                                   hidden_mults, activation = activation,
                                   complex = complex)
            if attention_embed:
                self.deembedder = CrossAttentionEmbedder(hidden_dim,
                                                         input_dim,
                                                         in_context = embed_mult,
                                                         out_context = delay,
                                                         num_layers = 1,
                                                         num_heads = 1)
            else:
                self.deembedder = torch.nn.Linear(hidden_dim * embed_mult, input_dim * delay)

        if self.neural_koopman:
            self.koopman = torch.nn.Parameter(torch.eye(bottleneck_dim,
                                                        dtype = torch.complex64 if complex else None))
            
    def decode(self, x):
        batch_size = x.shape[0]
        x_hat = self.decoder(x)

        if self.complex:
            x_hat = torch.view_as_real(x_hat)
            if self.attention_embed:
                x_hat = x_hat.transpose(1, 2)
            else:
                x_hat = x_hat.view(batch_size, -1)

        x_hat = self.deembedder(x_hat)
        x_hat = x_hat.reshape(batch_size, self.delay, -1)
        return x_hat

    def forward(self, x):
        batch_size = x.shape[0]
        if not self.attention_embed:
            x = x.view(batch_size, -1)
        x = self.embedder(x)

        if self.complex:
            # reshape and view as complex numbers
            x = x.reshape(batch_size, -1, 2).contiguous()
            x = torch.view_as_complex(x)

        x = self.encoder(x)
        if self.use_decoder:
            x_hat = self.decode(x)
        else:
            x_hat = None

        return x, x_hat
    
    def get_koopman(self, y_t, y_t1):
        """
        Given a pair of sequences y_t and y_t1, computes the Koopman operator.

        Parameters
        ----------
        y_t : torch.Tensor
            The first sequence of the pair.
        y_t1 : torch.Tensor
            The second sequence of the pair. Should be one step delayed from y_t.
        """
        if self.neural_koopman:
            A = self.koopman.clone()
            y_t1_hat = torch.einsum("ij,bj->bi",
                                    self.koopman,
                                    y_t)
            loss = (y_t1 - y_t1_hat).abs().mean()

        else:
            y_t_inv = torch.pinverse(y_t)
            A = torch.einsum("bdi,bjd->bij", y_t1, y_t_inv)
            Ay = torch.einsum("bij, bdj -> bdi", A, y_t)
            # frobenius norm of y_t1 - Ay
            loss = torch.linalg.matrix_norm(y_t1 - Ay).mean()
        return A, loss
    
    def get_loss(self, x_t, x_t1):
        """
        Given a pair of sequences x_t and x_t1, computes the loss of the model.

        #TODO write more here

        Parameters
        ----------
        x_t : torch.Tensor
            The first sequence of the pair.
        x_t1 : torch.Tensor
            The second sequence of the pair. Should be one step delayed from x_t.
        """
        with torch.no_grad():
            y_t, _ = self.forward(x_t)
        y_t1, x_t1_hat = self.forward(x_t1)

        _, loss = self.get_koopman(y_t, y_t1)

        if self.use_decoder:
            rec_loss = mse_loss(x_t1, x_t1_hat)
            loss += self.alpha * rec_loss
        return loss


#TODO : cleanup below
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from data_stream import FitzHughNagumoDS, TimeSpiralDS, train_on_ds
    DELAY = 8
    N_STEPS = 2000
    BATCH_SIZE = 512
    HIDDEN_DIM = 64
    HIDDEN_MULTS = [2, 2, 2, 1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LKIS(2,
                 DELAY,
                 HIDDEN_DIM * DELAY,
                 hidden_mults = HIDDEN_MULTS,
                 use_decoder = True)
    ds = TimeSpiralDS()

    losses = train_on_ds(model, ds,
                         n_steps = N_STEPS,
                         delay = DELAY,
                         batch_size = BATCH_SIZE)

    with torch.no_grad():
        sample = ds.sample(T = (DELAY + 1) * BATCH_SIZE,
                           batch_size = 1).to(device)
        sample = sample.view(BATCH_SIZE, DELAY + 1, 2)
        x_t = sample[:, :-1, :].detach().clone()
        x_t1 = sample[:, 1:, :].detach().clone()

        y_t, x_t_hat = model.forward(x_t)
        y_t1, x_t1_hat  = model.forward(x_t1)

        # this is the Koopman operator
        A, _ = model.get_koopman(y_t, y_t1)

        x_unseen = ds.sample(T = DELAY * (BATCH_SIZE // 2),
                           batch_size = 1).to(device)
        x_unseen = x_unseen.view(BATCH_SIZE // 2, DELAY, 2)
        y_unseen, x_unseen_hat = model.forward(x_unseen)
        y_unseen_hat = torch.einsum("ij, bj -> bi", A, y_unseen)

        x_future_hat = model.decode(y_unseen_hat)

        x_t_plt = x_unseen.view(DELAY * BATCH_SIZE // 2, 2)
        x_t_plt = x_t_plt.detach().cpu().numpy()
        x_t_plt_hat = x_unseen_hat.view(DELAY * BATCH_SIZE // 2, 2)
        x_t_plt_hat = x_t_plt_hat.detach().cpu().numpy()
        x_future_hat = x_future_hat.view(DELAY * BATCH_SIZE // 2, 2)
        x_future_hat = x_future_hat.detach().cpu().numpy()
        
    log_losses = np.log(losses)
    fig, ax = plt.subplots(2, 2,
                           figsize = (10, 10))
    ax[0, 0].plot(log_losses)
    ax[0, 0].set_title("Log Loss")
    ax[1, 0].imshow(A.real.detach().cpu().numpy())
    ax[1, 0].set_title("Koopman Matrix Approximation")
    ax[1, 1].imshow(A.imag.detach().cpu().numpy())

    #TODO : this is totally off
    ax[0, 1].plot(x_t_plt[:, 0],
               x_t_plt[:, 1],
               label = "$x_t$",
               alpha = 0.8)
    ax[0, 1].plot(x_future_hat[:, 0],
               x_future_hat[:, 1],
               label = "$\hat{x_t}$",
               alpha = 0.5)
    ax[0, 1].plot(x_t_plt_hat[:, 0],
                  x_t_plt_hat[:, 1],
                  label = "$\hat{x_t}$ (trained)",
                  alpha = 0.5)
    ax[0, 1].scatter(x_t_plt[0, 0],
                  x_t_plt[0, 1],
                  label = "Initial Point",
                  color = "red",
                  marker = "x")
    ax[0, 1].set_title("$x_t$ vs $\hat{x_t}$")

    ax[0, 1].legend()
    plt.tight_layout()



