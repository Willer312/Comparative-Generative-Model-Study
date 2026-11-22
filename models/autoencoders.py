import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaAutoencoder(nn.Module):
    """
    Deterministic Vanilla Autoencoder (784 -> 256 -> 64 -> 256 -> 784).
    Trained strictly with pixel-wise reconstruction loss (MSE).
    """
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.view(x.size(0), -1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    @staticmethod
    def loss_function(recon_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(recon_x, x)


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder (784 -> 256 -> 784) with L1 sparsity regularization
    on the hidden latent representation h: L = MSE(x_hat, x) + lambda * ||h||_1.
    """
    def __init__(self, input_dim: int = 784, latent_dim: int = 256, sparsity_lambda: float = 1e-4):
        super().__init__()
        self.latent_dim = latent_dim
        self.sparsity_lambda = sparsity_lambda
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.view(x.size(0), -1))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(h).view(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor):
        h = self.encode(x)
        recon = self.decode(h)
        return recon, h

    def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        recon_loss = F.mse_loss(recon_x, x)
        l1_penalty = torch.mean(torch.abs(h))
        return recon_loss + self.sparsity_lambda * l1_penalty


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE) with Gaussian posterior q(z|x) = N(mu, diag(sigma^2))
    and standard normal prior p(z) = N(0, I).
    """
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, latent_dim: int = 64, beta: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        self.fc_enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor):
        h = self.fc_enc(x.view(x.size(0), -1))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        self.eval()
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        bce = F.binary_cross_entropy(recon_x.view(batch_size, -1), x.view(batch_size, -1), reduction="sum") / batch_size
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        return bce + self.beta * kld


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand(8, 1, 28, 28, device=device)

    vae_vanilla = VanillaAutoencoder().to(device)
    r1, z1 = vae_vanilla(x)
    print("Vanilla AE:", r1.shape, "| Loss:", vae_vanilla.loss_function(r1, x).item())

    sae = SparseAutoencoder().to(device)
    r2, h2 = sae(x)
    print("Sparse AE: ", r2.shape, "| Loss:", sae.loss_function(r2, x, h2).item())

    vae = VariationalAutoencoder().to(device)
    r3, mu, logvar = vae(x)
    print("VAE:       ", r3.shape, "| Loss:", vae.loss_function(r3, x, mu, logvar).item())
    print("VAE Sample:", vae.sample(8, device).shape)