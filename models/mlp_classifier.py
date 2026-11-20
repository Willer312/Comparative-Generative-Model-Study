import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    """
    Stage 1: Supervised MLP Classifier mirroring an Autoencoder encoder structure
    (784 -> 256 -> 64 -> 10) for proving discriminative bottlenecks and non-invertible weights.
    """
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, latent_dim: int = 64, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)    
        self.fc2 = nn.Linear(hidden_dim, latent_dim) 
        self.fc3 = nn.Linear(latent_dim, num_classes)  

    def extract_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts penultimate latent features z_MLP in R^64."""
        x_flat = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x_flat))
        z_mlp = F.relu(self.fc2(h1))
        return z_mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_mlp = self.extract_latent(x)
        logits = self.fc3(z_mlp)
        return logits

    @torch.no_grad()
    def project_from_class_onehot(self, y_onehot: torch.Tensor) -> torch.Tensor:
        """
        Stage 2: Top-Down Weight Transposition from one-hot class vector y_c in R^10.
        Formula: x_hat = Sigmoid(W1^T * ReLU(W2^T * ReLU(W3^T * y_c)))
        Note: PyTorch nn.Linear stores weight as (out_features, in_features), so
        W^T @ v corresponds to F.linear(v, weight.t()).
        """
        self.eval()
        z_hat = F.relu(F.linear(y_onehot, self.fc3.weight.t()))
        h1_hat = F.relu(F.linear(z_hat, self.fc2.weight.t()))
        x_hat = torch.sigmoid(F.linear(h1_hat, self.fc1.weight.t()))
        return x_hat.view(-1, 1, 28, 28)

    @torch.no_grad()
    def project_from_latent(self, z_mlp: torch.Tensor) -> torch.Tensor:
        """
        Stage 3: Latent Transpose Projection from extracted test features z_MLP in R^64.
        Formula: x_hat = Sigmoid(W1^T * ReLU(W2^T * z_MLP))
        """
        self.eval()
        h1_hat = F.relu(F.linear(z_mlp, self.fc2.weight.t()))
        x_hat = torch.sigmoid(F.linear(h1_hat, self.fc1.weight.t()))
        return x_hat.view(-1, 1, 28, 28)


class FrozenLatentDecoder(nn.Module):
    """
    Stage 4: Separate MLP Decoder (64 -> 256 -> 784) trained with MSE Loss
    on frozen z_MLP features to test how much intra-class spatial detail was discarded.
    """
    def __init__(self, latent_dim: int = 64, hidden_dim: int = 256, output_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        return out.view(-1, 1, 28, 28)


@torch.no_grad()
def sample_mlp_prior(
    mlp_classifier: MLPClassifier,
    frozen_decoder: FrozenLatentDecoder,
    test_loader: torch.utils.data.DataLoader,
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Stage 5: Computes empirical mean (mu_z) and std (sigma_z) of z_MLP over the test set,
    samples z_rand ~ N(mu_z, diag(sigma_z^2)), and decodes via Stage 4's FrozenLatentDecoder.
    """
    mlp_classifier.eval()
    frozen_decoder.eval()

    latents = []
    for x, _ in test_loader:
        x = x.to(device)
        z = mlp_classifier.extract_latent(x)
        latents.append(z)

    all_z = torch.cat(latents, dim=0)  
    mu_z = all_z.mean(dim=0, keepdim=True)
    sigma_z = all_z.std(dim=0, keepdim=True) + 1e-6

    eps = torch.randn(num_samples, all_z.size(1), device=device)
    z_rand = mu_z + sigma_z * eps
    return frozen_decoder(z_rand)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf = MLPClassifier().to(device)
    dec = FrozenLatentDecoder().to(device)

    dummy_x = torch.rand(8, 1, 28, 28, device=device)
    logits = clf(dummy_x)
    z = clf.extract_latent(dummy_x)
    y_oh = F.one_hot(torch.arange(8, device=device), num_classes=10).float()

    print("Logits shape:", logits.shape)
    print("Latent z_MLP shape:", z.shape)
    print("Stage 2 class projection shape:", clf.project_from_class_onehot(y_oh).shape)
    print("Stage 3 latent projection shape:", clf.project_from_latent(z).shape)
    print("Stage 4 frozen decoder shape:", dec(z).shape)