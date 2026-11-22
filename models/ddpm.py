import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """Encodes scalar timestep t into a continuous vector embedding."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConvBlock(nn.Module):
    """Residual-style Double Conv block with GroupNorm, SiLU, and time-embedding injection."""
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        time_cond = self.act(self.time_mlp(t_emb))[:, :, None, None]
        h = h + time_cond
        h = self.act(self.norm2(self.conv2(h)))
        return h


class SimpleUNet(nn.Module):
    """
    Lightweight U-Net noise predictor eps_theta(x_t, t) for 28x28 grayscale images.
    Spatial path: 28x28 -> 14x14 -> 7x7 -> 14x14 -> 28x28 with skip connections.
    """
    def __init__(self, in_channels: int = 1, base_ch: int = 64, time_emb_dim: int = 128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

  
        self.enc1 = ConvBlock(in_channels, base_ch, time_emb_dim)        # 28x28
        self.down1 = nn.Conv2d(base_ch, base_ch, kernel_size=4, stride=2, padding=1)  # 14x14

        self.enc2 = ConvBlock(base_ch, base_ch * 2, time_emb_dim)        # 14x14
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, kernel_size=4, stride=2, padding=1)  # 7x7


        self.bottleneck = ConvBlock(base_ch * 2, base_ch * 4, time_emb_dim)  # 7x7


        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=4, stride=2, padding=1)  # 14x14
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2, time_emb_dim)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=4, stride=2, padding=1)  # 28x28
        self.dec1 = ConvBlock(base_ch * 2, base_ch, time_emb_dim)

        self.out_conv = nn.Conv2d(base_ch, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)

        # Down
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down1(e1), t_emb)

        # Middle
        b = self.bottleneck(self.down2(e2), t_emb)

        # Up
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb)

        return self.out_conv(d1)


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (Ho et al., 2020).
    Operates on images normalized to [-1, 1] with T=300 diffusion timesteps.
    """
    def __init__(self, timesteps: int = 300, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.timesteps = timesteps
        self.unet = SimpleUNet(in_channels=1, base_ch=64, time_emb_dim=128)

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """Forward diffusion process: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    def loss_function(self, x_0: torch.Tensor) -> torch.Tensor:
        """Supervised noise prediction objective: MSE(eps, eps_theta(x_t, t))."""
        batch_size = x_0.size(0)
        t = torch.randint(0, self.timesteps, (batch_size,), device=x_0.device).long()
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = self.unet(x_t, t)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t_index: int) -> torch.Tensor:
        """Single reverse denoising step: p_theta(x_{t-1} | x_t)."""
        batch_size = x_t.size(0)
        t_tensor = torch.full((batch_size,), t_index, device=x_t.device, dtype=torch.long)

        beta_t = self.betas[t_index]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t_index]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t_index]

        pred_noise = self.unet(x_t, t_tensor)
        model_mean = sqrt_recip_alpha_t * (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * pred_noise)

        if t_index == 0:
            return model_mean
        else:
            posterior_var_t = self.posterior_variance[t_index]
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device, return_trajectory: bool = False, num_steps_to_record: int = 8):
        """
        Iterative reverse diffusion sampling starting from x_T ~ N(0, I).
        If return_trajectory=True, also returns intermediate snapshots for Figure 5.
        """
        self.eval()
        x = torch.randn(num_samples, 1, 28, 28, device=device)

        record_indices = set(
            torch.linspace(self.timesteps - 1, 0, steps=num_steps_to_record).long().tolist()
        )
        trajectory = []

        for t_idx in reversed(range(self.timesteps)):
            x = self.p_sample(x, t_idx)
            if return_trajectory and t_idx in record_indices:
                trajectory.append((t_idx, x.detach().cpu()))

        x = torch.clamp(x, -1.0, 1.0)
        if return_trajectory:
            return x, trajectory
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ddpm = DDPM(timesteps=300).to(device)
    dummy_x0 = torch.randn(8, 1, 28, 28, device=device)
    loss = ddpm.loss_function(dummy_x0)
    samples, traj = ddpm.sample(4, device, return_trajectory=True, num_steps_to_record=6)
    print("DDPM Loss:", loss.item(), "| Sample shape:", samples.shape, "| Trajectory steps recorded:", len(traj))