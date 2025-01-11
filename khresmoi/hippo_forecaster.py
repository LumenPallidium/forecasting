import torch
from torch.special import legendre_polynomial_p
try:
    from hippo import legs_function_approx_trapezoidal as legs
    HIPPO_AVAILABLE = True

except ImportError:
    from warnings import warn
    warn("hippo not found, using slow version")
    HIPPO_AVAILABLE = False

class HiPPOLegS(torch.nn.Module):
    def __init__(self,
                 coef_dim,
                 length,
                 n_channels = 1,
                 batch_size = 1,
                 use_cpp = True,):
        super().__init__()
        self.coef_dim = coef_dim
        self.length = length
        self.n_channels = n_channels
        self.batch_size = batch_size

        self.use_cpp = use_cpp

        self.curr_t = None
        A, B, L = self._initialize_weights()
        self.register_buffer("A", A)
        self.register_buffer("B", B)
        self.register_buffer("L", L)
        self.register_buffer("coef", torch.zeros(self.batch_size, self.n_channels, self.coef_dim))

    def _initialize_weights(self):
        A = torch.diag(torch.arange(1,
                                    self.coef_dim + 1,
                                    dtype=torch.float32))
        
        B = torch.arange(0, self.coef_dim, dtype=torch.float32)
        B = torch.sqrt(2*B + 1)

        outer_product = torch.einsum("i,j->ij", B, B)
        A += outer_product.tril(diagonal=-1)

        L = []
        x_range = torch.linspace(-1, 1, self.length)
        for i in range(self.coef_dim):
            l = legendre_polynomial_p(x_range, i)
            sqrt_ = torch.sqrt(torch.tensor(2 * i + 1))
            L.append(l * sqrt_)
        L = torch.stack(L, dim=0)
        return A, B, L
    
    def forward(self, length = None):
        """
        Reconstuct the signal at times t.
        """
        if length is None:
            length = self.length
        out = torch.einsum("bci,ij->bcj", self.coef, self.L)

        return out[:, :, :length]

    def fit_coef(self, f_t, t_space):
        if HIPPO_AVAILABLE and self.use_cpp:
            f_t = f_t.reshape(self.batch_size * self.n_channels, -1)
            c_k = [legs(f_t[i, :], self.coef_dim) for i in range(f_t.shape[0])]
            c_k = torch.stack(c_k, dim=0)
            c_k = c_k.reshape(self.batch_size, self.n_channels, self.coef_dim)
            t = t_space[-1]
        else:
            c_k = self.coef.clone()
            for k in range(f_t.shape[-1]):
                t = t_space[k]
                k_inv = 1 / (k + 1)
                b_term = k_inv * torch.einsum("i,bc->bci", self.B, f_t[..., k])
                step_A = torch.einsum("ij,bcj->bci", self.A, c_k)
                c_k = c_k - (k_inv * step_A) + b_term
                #print(c_k.norm())

        self.coef = c_k
        self.curr_t = t
        return self.coef.clone()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # note that the slow version collapses for coef_dim >~ 32
    coef_dim = 32
    n_channels = 2
    len_ = 1024

    hippo_model = HiPPOLegS(coef_dim,
                            len_,
                            use_cpp = True,
                            n_channels = n_channels)
    hippo_model_slow = HiPPOLegS(coef_dim,
                                 len_,
                                 use_cpp = False,
                                 n_channels = n_channels)

    t_space = torch.linspace(0, 2, len_)
    simple_signal = torch.zeros(n_channels, len_)
    for i in range(10):
        simple_signal += torch.randn(n_channels)[:, None] * torch.sin(i * torch.pi * t_space)[None, :]
    rand_walk = (torch.randn_like(simple_signal) * 1).cumsum(-1)
    simple_signal += rand_walk
    simple_signal = simple_signal[None, ...]

    signal_np = simple_signal.cpu().numpy()
    plt.plot(t_space, signal_np[0, 0, :],
             label = "Signal")

    hippo_model.fit_coef(simple_signal, t_space)
    hippo_model_slow.fit_coef(simple_signal, t_space)

    approx = hippo_model()
    approx_np = approx.cpu().numpy()
    plt.plot(t_space, approx_np[0, 0, :],
             label = "HiPPOLegS")

    approx_slow = hippo_model_slow()
    approx_slow_np = approx_slow.cpu().numpy()
    plt.plot(t_space, approx_slow_np[0, 0, :],
             label = "HiPPOLegS Slow")
    plt.legend()
    