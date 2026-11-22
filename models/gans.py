import torch
import torch.nn as nn


def weights_init_dcgan(m: nn.Module) -> None:
    """Applies standard DCGAN normal weight initialization (mean=0.0, std=0.02)."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class VanillaGenerator(nn.Module):
    def __init__(self, latent_dim: int = 64, output_dim: int = 784):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, output_dim),
            nn.Tanh(),  
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, 1, 28, 28)

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        self.eval()
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.forward(z)


class VanillaDiscriminator(nn.Module):
    def __init__(self, input_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1), 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.size(0), -1)
        return self.net(x_flat)


class VanillaGAN(nn.Module):
    """Wrapper bundling Vanilla Generator & Discriminator with unified sampling."""
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = VanillaGenerator(latent_dim=latent_dim)
        self.discriminator = VanillaDiscriminator()

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        return self.generator.sample(num_samples, device)



class DCGANGenerator(nn.Module):
    """
    Maps z in R^100 -> (1, 28, 28) via strided transposed convolutions:
    (100, 1, 1) -> (256, 7, 7) -> (128, 14, 14) -> (1, 28, 28).
    """
    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(

            nn.ConvTranspose2d(latent_dim, 256, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),


            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),


            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(), 
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_4d = z.view(z.size(0), self.latent_dim, 1, 1)
        return self.net(z_4d)

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        self.eval()
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.forward(z)


class DCGANDiscriminator(nn.Module):
    """
    Maps (1, 28, 28) -> scalar logit via strided convolutions (no pooling):
    (1, 28, 28) -> (64, 14, 14) -> (128, 7, 7) -> (1, 1, 1).
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(

            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),


            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),


            nn.Conv2d(128, 1, kernel_size=7, stride=1, padding=0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, 1)


class DCGAN(nn.Module):
    """Wrapper bundling DCGAN Generator & Discriminator with DCGAN weight init."""
    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = DCGANGenerator(latent_dim=latent_dim)
        self.discriminator = DCGANDiscriminator()
        self.apply(weights_init_dcgan)

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        return self.generator.sample(num_samples, device)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vgan = VanillaGAN(latent_dim=64).to(device)
    fake_v = vgan.sample(8, device)
    d_v = vgan.discriminator(fake_v)
    print("Vanilla GAN sample:", fake_v.shape, "| D logits:", d_v.shape)

    dcgan = DCGAN(latent_dim=100).to(device)
    fake_dc = dcgan.sample(8, device)
    d_dc = dcgan.discriminator(fake_dc)
    print("DCGAN sample:      ", fake_dc.shape, "| D logits:", d_dc.shape)