import torch
from modules import MLP, Transformer, FunGen

MEASUREMENT_FUNCTIONS = [torch.sin, torch.cos, torch.exp,
                         torch.sigmoid, torch.tanh]

class KNFEncoder(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 num_steps,
                 measurement_functions = MEASUREMENT_FUNCTIONS):
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
        Input shape is (batch_size, n_chunks, num_steps, in_dim)
        """
        # (batch_size, num_steps, in_dim) -> (batch_size, n_measurements, num_steps, in_dim)
        x_mod = torch.einsum("...dj,ingdj->...ing", x, self.encoder_matrix)
        # (batch_size, n_measurements, num_steps, in_dim) -> (batch_size, n_measurements, in_dim)
        v = torch.einsum("...nlj,...lj->...nj", x_mod, x)
        v = torch.stack([f(v[..., i, :]) for i, f in enumerate(self.measurement_functions)], dim=-2)
        return v # (batch_size, n_chunks, n_measurements, in_dim)
    
class FunGenEncoder(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 num_steps,
                 n_measurements):
        super().__init__()
        self.in_dim = in_dim
        self.num_steps = num_steps
        self.n_measurements = n_measurements

        self.fungen = FunGen(in_dim * num_steps, in_dim * self.n_measurements)

    def forward(self, x):
        """
        Input shape is (batch_size, n_chunks, num_steps, in_dim)
        """
        x = x.view(x.shape[0], x.shape[1], -1)
        v = self.fungen(x)
        v = v.view(x.shape[0], x.shape[1], self.n_measurements, self.in_dim)
        return v # (batch_size, n_chunks, n_measurements, in_dim)
    
class KNFDecoder(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 num_steps,
                 n_measurements):
        super().__init__()
        self.in_dim = in_dim
        self.num_steps = num_steps
        self.n_measurements = n_measurements

        self.decoder_matrix = torch.nn.Parameter(torch.randn(num_steps,
                                                             in_dim,
                                                             self.n_measurements,
                                                             in_dim) * 1e-2)

    def forward(self, v):
        x = torch.einsum("...nd,ijnd->...ij",
                         v,
                         self.decoder_matrix)
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
                 measurement_functions = MEASUREMENT_FUNCTIONS,
                 n_measurements = 5,
                 use_lookback = True,
                 use_fun_gen = False):
        super().__init__()
        self.in_dim = in_dim
        self.num_steps = num_steps
        # dividing here makes things easier
        self.lookback_steps = lookback_steps
        
        if measurement_functions is not None:
            self.measurement_functions = measurement_functions
            n_measurements = len(measurement_functions)

        self.n_measurements = n_measurements
        self.dim = in_dim * self.n_measurements

        self.use_fun_gen = use_fun_gen
        if use_fun_gen:
            self.encoder = FunGenEncoder(in_dim,
                                         num_steps,
                                         n_measurements)
        else:
            self.encoder = KNFEncoder(in_dim, num_steps,
                                      measurement_functions)
        self.decoder = KNFDecoder(in_dim, num_steps, self.n_measurements)

        self.koopman_global = torch.nn.Parameter(torch.randn(self.dim,
                                                             self.dim,) * 1e-2)
        self.koopman_local = Transformer(self.in_dim * self.n_measurements,
                                         transformer_layers,
                                         context = self.lookback_steps,
                                         causal = True,
                                         heads = 2,
                                         dropout = 0.1)
        if use_lookback:
            self.koopman_lookback = MLP(self.in_dim * self.n_measurements,
                                        2 * self.in_dim * self.n_measurements)
            self.use_lookback = use_lookback
        else:
            self.use_lookback = False
        
    #TODO maybe add complex dtype but that is hard with the transformer
    def forward(self, x, return_aux = False):
        batch_size, total_steps, dim = x.shape
        n_chunks = total_steps // self.num_steps
        #TODO pad?
        x = x.view(batch_size, n_chunks, self.num_steps, dim)

        steps = n_chunks // self.lookback_steps
        v_full = self.encoder(x).view(batch_size,
                                      steps,
                                      self.lookback_steps,
                                      -1)

        chunks = []
        # for loop because lookback network is recursive
        for chunk in range(0, steps):
            v = v_full[:, chunk, :]
            v_hat_global = torch.einsum("...j,ji->...i",
                                        v,
                                        self.koopman_global)
            
            v_hat_local = self.koopman_local(v)

            v_hat = v_hat_global + v_hat_local
            lookback_loss = 0
            if self.use_lookback & (chunk != steps - 1):
                next_chunk = v_full[:, chunk + 1, :]
                next_chunk = next_chunk.detach()
                pred_error = v_hat - next_chunk

                lookback_loss += (pred_error ** 2).sum()
                v_hat_lookback = self.koopman_lookback(pred_error)
                v_hat = v_hat + v_hat_lookback

            v_hat = v_hat.view(batch_size,
                               self.lookback_steps,
                               self.n_measurements,
                               self.in_dim)

            chunks.append(v_hat)

        v_hat = torch.stack(chunks, dim = 1)

        x_hat = self.decoder(v_hat)
        x_hat = x_hat.view(batch_size, -1, self.in_dim)
        if return_aux:
            v_full = v_full.view(batch_size,
                                 n_chunks,
                                 self.n_measurements,
                                 self.in_dim)
            recons = self.decoder(v_full).view(batch_size,
                                               total_steps,
                                               dim)
            return x_hat, recons, lookback_loss
        return x_hat
    
    def get_loss(self, x_t, x_t1):
        x_t1_hat, recons, l_lookback = self(x_t, return_aux = True)
        l_rec = torch.nn.functional.mse_loss(recons, x_t)
        l_pred = torch.nn.functional.mse_loss(x_t1_hat, x_t1)
        return l_rec + l_pred + l_lookback
        
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from data_stream import FitzHughNagumoDS, train_on_ds
    DELAY = 4
    LOOKBACK_DELAY = 8
    N_CHUNKS = 32
    N_STEPS = 1000
    BATCH_SIZE = 512
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = KNF(2, DELAY, LOOKBACK_DELAY)
    model_no_lookback = KNF(2, DELAY, LOOKBACK_DELAY, use_lookback = False)
    model_fun_gen = KNF(2, DELAY, LOOKBACK_DELAY, use_fun_gen = True)
    model_fun_gen_no_lookback = KNF(2, DELAY, LOOKBACK_DELAY, use_lookback = False, use_fun_gen = True)
    ds = FitzHughNagumoDS()

    losses = train_on_ds(model, ds,
                         n_steps = N_STEPS,
                         delay = DELAY * N_CHUNKS,
                         batch_size = BATCH_SIZE)
    losses_no_lookback = train_on_ds(model_no_lookback, ds,
                                     n_steps = N_STEPS,
                                     delay = DELAY * N_CHUNKS,
                                     batch_size = BATCH_SIZE)
    losses_fun_gen = train_on_ds(model_fun_gen, ds,
                                 n_steps = N_STEPS,
                                 delay = DELAY * N_CHUNKS,
                                 batch_size = BATCH_SIZE)
    losses_fun_gen_no_lookback = train_on_ds(model_fun_gen_no_lookback, ds,
                                             n_steps = N_STEPS,
                                             delay = DELAY * N_CHUNKS,
                                             batch_size = BATCH_SIZE)
    
    smooth_losses = np.convolve(losses, np.ones(100) / 100, mode = "valid")
    smooth_losses_no_lookback = np.convolve(losses_no_lookback, np.ones(100) / 100, mode = "valid")
    smooth_losses_fun_gen = np.convolve(losses_fun_gen, np.ones(100) / 100, mode = "valid")
    smooth_losses_fun_gen_no_lookback = np.convolve(losses_fun_gen_no_lookback, np.ones(100) / 100, mode = "valid")

    model_fun_gen.eval()
    test_traj = ds.sample(T = 10 * (DELAY *  LOOKBACK_DELAY),
                          batch_size = 1).to(device)
    x = test_traj[:, :(DELAY *  LOOKBACK_DELAY), :]
    preds = [x]
    for i in range(1, 10):
        x = model_fun_gen(x)
        preds.append(x[:, -DELAY:, :])
    preds = torch.cat(preds, dim = 1).detach().cpu().numpy()


    fig, ax = plt.subplots(1, 3, figsize = (10, 5))
    ax[0].plot(smooth_losses,
               label = "Full")
    ax[0].plot(smooth_losses_no_lookback,
               label = "No Lookback")
    ax[0].plot(smooth_losses_fun_gen,
               label = "Fun Gen")
    ax[0].plot(smooth_losses_fun_gen_no_lookback,
               label = "Fun Gen, No Lookback")
    ax[0].set_title("Loss")
    ax[0].legend()
    A = model.koopman_global.detach().cpu().numpy()
    ax[1].imshow(A)
    ax[1].set_title("Koopman Matrix Approximation")

    ax[2].plot(test_traj[0, :, 0].cpu().numpy(),
               test_traj[0, :, 1].cpu().numpy(),
               label = "True")
    ax[2].plot(preds[0, :, 0],
                preds[0, :, 1],
                label = "Pred")
    
    ax[2].set_title("Trajectory")
    ax[2].legend()
