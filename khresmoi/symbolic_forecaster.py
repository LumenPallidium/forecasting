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
                 quantizer_loss_weight = 1e-3):
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
        # normalize rows
        markov_matrix /= (markov_matrix.sum(dim = -1, keepdim = True) + 1e-8)
        # get row entropy
        entropy = -torch.sum(markov_matrix * torch.log(markov_matrix + 1e-8), dim = -1).mean()
        loss = entropy * self.entropy_loss_weight + inner_loss_t * self.quantizer_loss_weight
        return torch.nn.functional.mse_loss(x_t_recons, x_t) + loss


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from data_stream import FitzHughNagumoDS, train_on_ds
    DELAY = 16
    N_STEPS = 1000
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
    fig, ax = plt.subplots(figsize = (10, 5))
    ax.plot(log_losses)
    ax.set_title("Log Loss")