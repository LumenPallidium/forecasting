import torch
from som_quantize.quantizer import ResidualQuantizer

class Symbolic(torch.nn.Module):
    def __init__(self,
                 dim,
                 encoder,
                 decoder,
                 codebook_size = 256,
                 residual_count = 2,
                 entropy_loss_weight = 0.1,
                 quantizer_loss_weight = 1e-3,
                 alpha = 0.5):
        super().__init__()
        self.encoder = encoder
        self.quantizer = ResidualQuantizer(residual_count,
                                           dim,
                                           codebook_size,
                                           use_som = False,
                                           probabilistic =True)
        self.decoder = decoder

        self.entropy_loss_weight = entropy_loss_weight
        self.quantizer_loss_weight = quantizer_loss_weight
        self.alpha = alpha

        self.register_buffer("markov_matrix", torch.ones(residual_count,
                                                         codebook_size,
                                                         codebook_size) / codebook_size ** 2)

    def forward(self, x):
        x_encoded = self.encoder(x)
        x_quantized, index_probs, inner_loss = self.quantizer(x_encoded)
        x_reconstructed = self.decoder(x_quantized)
        return x_reconstructed, index_probs, inner_loss
    
    def get_loss(self, x_t, x_t1):
        x_t_recons, index_probs_t, inner_loss_t = self(x_t)
        with torch.no_grad():
            x_t1_recons, index_probs_t1, inner_loss_t1 = self(x_t1)
        markov_matrix = torch.einsum("btic, btjc -> cij", index_probs_t, index_probs_t1)
        markov_matrix_no_diag = markov_matrix - torch.diag_embed(torch.diagonal(markov_matrix, dim1 = -2, dim2 = -1))
        # normalize rows
        markov_matrix_no_diag /= (markov_matrix_no_diag.sum(dim = -1, keepdim = True) + 1e-8)
        markov_matrix /= (markov_matrix.sum(dim = -1, keepdim = True) + 1e-8)
        self.markov_matrix.mul_(self.alpha).add_(markov_matrix, alpha = 1 - self.alpha)
        # get row entropy
        entropy = -torch.sum(markov_matrix_no_diag * torch.log(markov_matrix_no_diag + 1e-8), dim = -1).mean()
        loss = entropy * self.entropy_loss_weight + inner_loss_t * self.quantizer_loss_weight
        return torch.nn.functional.mse_loss(x_t_recons, x_t) + loss


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from data_stream import FitzHughNagumoDS, train_on_ds
    DELAY = 16
    N_STEPS = 100
    BATCH_SIZE = 512
    HIDDEN_DIM = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = torch.nn.Sequential(
        torch.nn.Linear(2, HIDDEN_DIM),
        torch.nn.GELU(),
        torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        torch.nn.GELU())
    decoder = torch.nn.Sequential(
        torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        torch.nn.GELU(),
        torch.nn.Linear(HIDDEN_DIM, 2))

    model = Symbolic(HIDDEN_DIM, encoder, decoder)
    ds = FitzHughNagumoDS()

    losses = train_on_ds(model, ds,
                         n_steps = N_STEPS,
                         delay = DELAY,
                         batch_size = BATCH_SIZE)
    
    log_losses = np.log(losses)
    fig, ax = plt.subplots(1, 3,figsize = (15, 5))
    ax[0].plot(log_losses)
    ax[0].set_title("Log Loss")

    # imshow markov matrix
    ax[1].imshow(model.markov_matrix[0].detach().cpu().numpy())
    ax[1].set_title("Markov Matrix")

    sample = ds.sample(T = 10000, dt= 0.1, batch_size = 1).squeeze(0).cpu().numpy()
    ax[2].plot(sample[:, 0], sample[:, 1])
    ax[2].set_title("Sample Trajectory")

    # decode first codebook
    codebook = model.quantizer.quantizers[0].codebook
    codebook_decoded = decoder(codebook.T).detach().cpu().numpy()
    ax[2].scatter(codebook_decoded[:, 0], codebook_decoded[:, 1], c = "red")
