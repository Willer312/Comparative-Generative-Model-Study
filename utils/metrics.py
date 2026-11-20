import time
import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance


def denormalize_to_zero_one(x: torch.Tensor, from_neg_one: bool = False) -> torch.Tensor:
    """Ensures image tensor is clamped strictly within [0, 1]."""
    if from_neg_one:
        x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)


@torch.no_grad()
def evaluate_reconstruction_metrics(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    is_mlp_frozen_decoder: bool = False,
    mlp_encoder: torch.nn.Module = None,
) -> dict:
    """
    Computes Test MSE, Test PSNR (dB), and Test SSIM over the entire test set.
    Used for Vanilla AE, Sparse AE, VAE, and the Stage 4 MLP Frozen Decoder.
    """
    model.eval()
    if mlp_encoder is not None:
        mlp_encoder.eval()

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    total_mse = 0.0
    total_samples = 0

    for x, _ in test_loader:
        x = x.to(device)
        batch_size = x.size(0)

        if is_mlp_frozen_decoder and mlp_encoder is not None:
            z_mlp = mlp_encoder.extract_latent(x)
            recon = model(z_mlp)
        else:
            out = model(x)
            # Handle models returning tuples like (recon, mu, logvar) or (recon, latent)
            recon = out[0] if isinstance(out, tuple) else out

        recon = recon.view_as(x)
        recon = torch.clamp(recon, 0.0, 1.0)

        # Accumulate sum of per-image MSE to compute exact dataset average
        batch_mse = F.mse_loss(recon, x, reduction="none").view(batch_size, -1).mean(dim=1).sum().item()
        total_mse += batch_mse
        total_samples += batch_size

        psnr_metric.update(recon, x)
        ssim_metric.update(recon, x)

    return {
        "mse": total_mse / total_samples,
        "psnr": psnr_metric.compute().item(),
        "ssim": ssim_metric.compute().item(),
    }


@torch.no_grad()
def compute_fid_score(
    real_loader: torch.utils.data.DataLoader,
    sample_fn,
    device: torch.device,
    num_samples: int = 2000,
    batch_size: int = 128,
    generated_is_neg_one: bool = False,
) -> float:
    """
    Computes Fréchet Inception Distance (FID) between real test set images
    and synthetic images generated from prior noise.
    
    Args:
        sample_fn: Callable taking (n_samples, device) and returning (N, 1, 28, 28) tensor.
        num_samples: Number of synthetic images to generate (2000 is fast and stable).
    """
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    def to_uint8_rgb(imgs: torch.Tensor) -> torch.Tensor:
        # imgs expected in [0, 1], shape (B, 1, 28, 28)
        imgs_3ch = imgs.repeat(1, 3, 1, 1)
        return (imgs_3ch * 255.0).clamp(0, 255).to(torch.uint8)

    # 1. Feed real test images (assumed in [0, 1] from standard test_loader)
    real_count = 0
    for x, _ in real_loader:
        x = x.to(device)
        fid.update(to_uint8_rgb(x), real=True)
        real_count += x.size(0)
        if real_count >= num_samples:
            break

    # 2. Feed generated fake images
    fake_count = 0
    while fake_count < num_samples:
        curr_batch = min(batch_size, num_samples - fake_count)
        fake_imgs = sample_fn(curr_batch, device)
        fake_imgs = denormalize_to_zero_one(fake_imgs, from_neg_one=generated_is_neg_one)
        fid.update(to_uint8_rgb(fake_imgs), real=False)
        fake_count += curr_batch

    return fid.compute().item()


@torch.no_grad()
def measure_inference_latency(
    inference_fn,
    device: torch.device,
    batch_size: int = 64,
    warmup_runs: int = 5,
    timed_runs: int = 20,
) -> float:
    """
    Measures average inference latency in milliseconds for a single batch of 64 images.
    """
    # Warmup GPU caches/kernels
    for _ in range(warmup_runs):
        _ = inference_fn(batch_size, device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(timed_runs):
        _ = inference_fn(batch_size, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
    end_time = time.perf_counter()

    avg_ms = ((end_time - start_time) / timed_runs) * 1000.0
    return avg_ms

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing metrics on device: {device}")

    # 1. Dummy loader with 128 grayscale images in [0, 1]
    dummy_imgs = torch.rand(128, 1, 28, 28)
    dummy_labels = torch.zeros(128, dtype=torch.long)
    dummy_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(dummy_imgs, dummy_labels), batch_size=64
    )

    # 2. Dummy identity model
    class DummyAE(torch.nn.Module):
        def forward(self, x):
            return torch.clamp(x + 0.05 * torch.randn_like(x), 0.0, 1.0)

    recon_res = evaluate_reconstruction_metrics(DummyAE().to(device), dummy_loader, device)
    print("Reconstruction test:", recon_res)

    # 3. Dummy sampler for FID & Latency
    def dummy_sampler(n, dev):
        return torch.rand(n, 1, 28, 28, device=dev)

    lat = measure_inference_latency(dummy_sampler, device, batch_size=64)
    print(f"Latency test (ms/batch 64): {lat:.3f} ms")

    fid_val = compute_fid_score(dummy_loader, dummy_sampler, device, num_samples=128, batch_size=64)
    print(f"FID test: {fid_val:.2f}")