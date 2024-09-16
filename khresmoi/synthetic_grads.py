import torch
from torch.func import jacrev, vmap
from torch.nn.functional import mse_loss

class GateRNN(torch.nn.Module):
    """
    Custom gated RNN for working with synthetic gradients.
    """
    def __init__(self,
                 in_dim,
                 dim,
                 activation = torch.nn.GELU()):
        super().__init__()
        self.in_dim = in_dim
        self.dim = dim

        self.Wx = torch.nn.Parameter(torch.randn(in_dim, dim) * 1e-2)
        self.Wgate = torch.nn.Parameter(torch.randn(dim, dim) * 1e-2)
        self.Wh = torch.nn.Parameter(torch.randn(dim, dim) * 1e-2)

        self.activation = activation
        self.hidden_grads = []
        self.register_buffer("dh_dWh", torch.zeros(dim, dim))

    
    def partial_forward(self, x, h, Wh):
        new_h = self.activation(x @ self.Wx + h @ Wh)
        gate = torch.sigmoid(new_h @ self.Wgate)
        h = gate * h + (1 - gate)
        return h, h
    
    def forward(self, x, h = None, take_grad = True):
        self.hidden_grads.clear()
        if h is None:
            h = torch.zeros(x.shape[0],
                            self.dim,
                            device=x.device,
                            requires_grad=True)
            
        dh_dh_func = jacrev(self.partial_forward,
                            1,
                            has_aux = True)
            
        n_steps = x.shape[1]
        y = []
        for i in range(n_steps):
            if take_grad:
                Wh = self.Wh.unsqueeze(0).repeat(x.shape[0], 1, 1)
                dh_dh, h = vmap(dh_dh_func)(x[:, i, :],
                                            h,
                                            Wh)
                self.hidden_grads.append(dh_dh)
                if (i == n_steps - 1) and take_grad:
                    dh_dWh_func = jacrev(self.partial_forward,
                                         2)
                    self.dh_dWh = dh_dWh_func(x[:, i, :],
                                              h,
                                              self.Wh)[0]
            else:
                h = self.partial_forward(x, h, i, self.Wh)
            y.append(h)
        
        y = torch.stack(y, dim=1)

        return h, y

class SynGradBlock(torch.nn.Module):
    def __init__(self,
                 dim):
        super().__init__()
        self.dim = dim

        self.rnn = GateRNN(dim, dim)
        self.synth = torch.nn.Linear(dim, dim)

    def forward(self, x, h = None):
        # expect shape N, L, D
        
        h, y = self.rnn(x, h)
        grad_hat = self.synth(h)

        return h, y, grad_hat
    
    def get_synthetic_grad(self, y):
        # note this assumes .backward() has been called
        all_hidden_grads = torch.stack(self.rnn.hidden_grads,
                                       dim = 1)
        synth_grad = torch.einsum("nld,de->nle",
                                  y,
                                  self.synth.weight)
        with torch.no_grad():
            bootstrap = torch.einsum("nld,nled->nle",
                                    synth_grad,
                                    all_hidden_grads)
        # bootstrap is one step ahead of synth_grad
        synth_loss = mse_loss(synth_grad[:, 1:, :],
                              bootstrap[:, :-1, :].detach())
        # TODO : check this, should it be matmul
        bootstrap_grad = torch.einsum("nld,ndwh->lwh",
                                      synth_grad,
                                      self.rnn.dh_dWh)
        bootstrap_grad = bootstrap_grad.mean(dim = 1) / y.shape[0]
        return synth_loss, bootstrap_grad
    
if __name__ == "__main__":
    block = SynGradBlock(10)
    x = torch.randn(5, 8, 10)
    h, y, grad_hat = block(x)
    print(y.shape, h.shape)
    loss = y.sum()
    loss.backward()

    synth_loss, synth_grad = block.get_synthetic_grad(y)
    print(synth_loss)
    print(synth_grad.shape)
        
