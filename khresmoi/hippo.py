import torch
from torch.special import legendre_polynomial_p

class HiPPOLegS(torch.nn.Module):
    def __init__(self, coef_dim):
        super().__init__()
        self.coef_dim = coef_dim
        self.curr_t = None
        A, B = self._initialize_weights(coef_dim)
        self.register_buffer("A", A)
        self.register_buffer("B", B)
        self.register_buffer("coef", torch.zeros(coef_dim))

    def _initialize_weights(self, coef_dim):
        A = torch.diag(torch.arange(1,
                                    coef_dim + 1,
                                    dtype=torch.float32))
        
        B = torch.arange(0, coef_dim, dtype=torch.float32)
        B = torch.sqrt(2*B + 1)

        outer_product = torch.einsum("i,j->ij", B, B)
        A += outer_product.tril(diagonal=-1)
        return A, B
    
    def forward(self, t):
        """
        Reconstuct the signal at times t.
        """
        out = torch.zeros_like(t)
        t_scaled = (2 * t / self.curr_t) - 1
        #TODO : can we vectorize this?
        for i in range(1, self.coef_dim + 1):
            sqrt_ = torch.sqrt(torch.tensor(2 * i + 1))
            lp = legendre_polynomial_p(t_scaled, i)

            out += self.coef[i - 1] * sqrt_* lp
        return out

    #TODO: this often diverges for high coef_dim
    def fit_coef(self, f_t, k, t):
        k_inv = 1 / k

        b_term = k_inv * self.B * f_t
        curr_A = self.A.clone()
        #curr_A[k:, k:] = 0

        c_k = self.coef - (k_inv * curr_A @ self.coef) + b_term

        #print(c_k.norm())

        self.coef = c_k
        self.curr_t = t

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    hippo = HiPPOLegS(32)

    t_space = torch.linspace(0, 10, 1000)
    simple_signal = torch.zeros_like(t_space)
    for i in range(10):
        simple_signal += torch.randn(1) * torch.sin(i * torch.pi * t_space)

    signal_np = simple_signal.cpu().numpy()
    plt.plot(t_space, signal_np)

    for k in range(simple_signal.shape[0]):
        t = t_space[k]
        hippo.fit_coef(simple_signal[k], k + 1, t)

    approx = hippo(t_space)
    approx_np = approx.cpu().numpy()
    plt.plot(t_space, approx_np)
    