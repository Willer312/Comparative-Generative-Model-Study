import argparse
import json
import os
import pandas as pd
import torch

from utils.seed import set_seed
from utils.dataset import get_fashion_mnist_loaders
from utils.metrics import (
    evaluate_reconstruction_metrics,
    compute_fid_score,
    measure_inference_latency,
)
from utils.visualize import (
    generate_figure_1_reconstructions,
    generate_figure_2_random_sampling,
    generate_figure_3_mlp_proof,
    generate_figure_4_latent_interpolation,
    generate_figure_5_losses_and_ddpm_trajectory,
)
from models.mlp_classifier import MLPClassifier, FrozenLatentDecoder
from models.autoencoders import VanillaAutoencoder, SparseAutoencoder, VariationalAutoencoder
from models.gans import VanillaGAN, DCGAN
from models.ddpm import DDPM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--fid_samples", type=int, default=2000)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs("outputs/figures", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device} | Seed: {args.seed}")

    _, _, test_loader = get_fashion_mnist_loaders(
        batch_size=args.batch_size, seed=args.seed, normalize_to_neg_one=False
    )

    history_path = "outputs/training_history.json"
    history = json.load(open(history_path)) if os.path.exists(history_path) else {}

    mlp_clf = MLPClassifier().to(device)
    mlp_clf.load_state_dict(torch.load("outputs/checkpoints/mlp_classifier.pth", map_location=device))

    frozen_dec = FrozenLatentDecoder().to(device)
    frozen_dec.load_state_dict(torch.load("outputs/checkpoints/mlp_frozen_decoder.pth", map_location=device))

    vanilla_ae = VanillaAutoencoder(latent_dim=64).to(device)
    vanilla_ae.load_state_dict(torch.load("outputs/checkpoints/vanilla_ae.pth", map_location=device))

    sparse_ae = SparseAutoencoder(latent_dim=256, sparsity_lambda=1e-4).to(device)
    sparse_ae.load_state_dict(torch.load("outputs/checkpoints/sparse_ae.pth", map_location=device))

    vae = VariationalAutoencoder(latent_dim=64, beta=1.0).to(device)
    vae.load_state_dict(torch.load("outputs/checkpoints/vae.pth", map_location=device))

    vanilla_gan = VanillaGAN(latent_dim=64).to(device)
    vanilla_gan.load_state_dict(torch.load("outputs/checkpoints/vanilla_gan.pth", map_location=device))

    dcgan = DCGAN(latent_dim=100).to(device)
    dcgan.load_state_dict(torch.load("outputs/checkpoints/dcgan.pth", map_location=device))

    ddpm = DDPM(timesteps=300).to(device)
    ddpm.load_state_dict(torch.load("outputs/checkpoints/ddpm.pth", map_location=device))

    print("\n--- Evaluating Part 2: MLP Classifier Frozen Decoder vs Vanilla AE ---")
    mlp_recon_metrics = evaluate_reconstruction_metrics(
        frozen_dec, test_loader, device, is_mlp_frozen_decoder=True, mlp_encoder=mlp_clf
    )
    vae_vanilla_metrics = evaluate_reconstruction_metrics(vanilla_ae, test_loader, device)
    print(f"MLP Classifier Test Acc: {history.get('mlp_test_acc', 0.0):.2f}%")
    print(f"Frozen MLP Decoder (z_MLP=64) -> MSE: {mlp_recon_metrics['mse']:.5f} | PSNR: {mlp_recon_metrics['psnr']:.2f} dB | SSIM: {mlp_recon_metrics['ssim']:.4f}")
    print(f"Vanilla AE         (z_AE=64)  -> MSE: {vae_vanilla_metrics['mse']:.5f} | PSNR: {vae_vanilla_metrics['psnr']:.2f} dB | SSIM: {vae_vanilla_metrics['ssim']:.4f}")

    print("\n--- Computing Quantitative Table IV Metrics ---")
    sparse_metrics = evaluate_reconstruction_metrics(sparse_ae, test_loader, device)
    vae_metrics = evaluate_reconstruction_metrics(vae, test_loader, device)

    dummy_64_img = torch.randn(64, 1, 28, 28, device=device)
    lat_vanilla_ae = measure_inference_latency(lambda bs, dev: vanilla_ae(dummy_64_img), device)
    lat_sparse_ae = measure_inference_latency(lambda bs, dev: sparse_ae(dummy_64_img), device)
    lat_vae = measure_inference_latency(lambda bs, dev: vae.sample(bs, dev), device)
    lat_vanilla_gan = measure_inference_latency(lambda bs, dev: vanilla_gan.sample(bs, dev), device)
    lat_dcgan = measure_inference_latency(lambda bs, dev: dcgan.sample(bs, dev), device)
    lat_ddpm = measure_inference_latency(lambda bs, dev: ddpm.sample(bs, dev), device, warmup_runs=1, timed_runs=3)

    print(f"Computing FID Scores across {args.fid_samples} synthetic samples...")
    fid_vae = compute_fid_score(test_loader, vae.sample, device, num_samples=args.fid_samples, generated_is_neg_one=False)
    fid_vanilla_gan = compute_fid_score(test_loader, vanilla_gan.sample, device, num_samples=args.fid_samples, generated_is_neg_one=True)
    fid_dcgan = compute_fid_score(test_loader, dcgan.sample, device, num_samples=args.fid_samples, generated_is_neg_one=True)
    fid_ddpm = compute_fid_score(test_loader, ddpm.sample, device, num_samples=args.fid_samples, generated_is_neg_one=True)

    table_rows = [
        {
            "Arsitektur Model": "Vanilla Autoencoder",
            "Dim Laten": "64",
            "Training Time / Epoch (s)": f"{history.get('vanilla_ae', {}).get('avg_epoch_time', 0):.2f}",
            "Latency (ms / batch 64)": f"{lat_vanilla_ae:.2f}",
            "Test MSE": f"{vae_vanilla_metrics['mse']:.5f}",
            "Test PSNR (dB)": f"{vae_vanilla_metrics['psnr']:.2f}",
            "Test SSIM": f"{vae_vanilla_metrics['ssim']:.4f}",
            "FID Score": "N/A (Latent Voids)",
            "Kestabilan Training": "Stabil / Konvergen",
        },
        {
            "Arsitektur Model": "Sparse Autoencoder",
            "Dim Laten": "256 (Sparse)",
            "Training Time / Epoch (s)": f"{history.get('sparse_ae', {}).get('avg_epoch_time', 0):.2f}",
            "Latency (ms / batch 64)": f"{lat_sparse_ae:.2f}",
            "Test MSE": f"{sparse_metrics['mse']:.5f}",
            "Test PSNR (dB)": f"{sparse_metrics['psnr']:.2f}",
            "Test SSIM": f"{sparse_metrics['ssim']:.4f}",
            "FID Score": "N/A (Latent Voids)",
            "Kestabilan Training": "Stabil / Konvergen",
        },
        {
            "Arsitektur Model": "Variational Autoencoder (VAE)",
            "Dim Laten": "64",
            "Training Time / Epoch (s)": f"{history.get('vae', {}).get('avg_epoch_time', 0):.2f}",
            "Latency (ms / batch 64)": f"{lat_vae:.2f}",
            "Test MSE": f"{vae_metrics['mse']:.5f}",
            "Test PSNR (dB)": f"{vae_metrics['psnr']:.2f}",
            "Test SSIM": f"{vae_metrics['ssim']:.4f}",
            "FID Score": f"{fid_vae:.2f}",
            "Kestabilan Training": "Stabil / Konvergen",
        },
        {
            "Arsitektur Model": "Generative Adversarial Network (GAN)",
            "Dim Laten": "64 (Prior Noise)",
            "Training Time / Epoch (s)": f"{history.get('vanilla_gan', {}).get('avg_epoch_time', 0):.2f}",
            "Latency (ms / batch 64)": f"{lat_vanilla_gan:.2f}",
            "Test MSE": "N/A (Tanpa Encoder)",
            "Test PSNR (dB)": "N/A",
            "Test SSIM": "N/A",
            "FID Score": f"{fid_vanilla_gan:.2f}",
            "Kestabilan Training": "Rentan Osilasi / Mode Collapse",
        },
        {
            "Arsitektur Model": "Deep Convolutional GAN (DCGAN)",
            "Dim Laten": "100 (Prior Noise)",
            "Training Time / Epoch (s)": f"{history.get('dcgan', {}).get('avg_epoch_time', 0):.2f}",
            "Latency (ms / batch 64)": f"{lat_dcgan:.2f}",
            "Test MSE": "N/A (Tanpa Encoder)",
            "Test PSNR (dB)": "N/A",
            "Test SSIM": "N/A",
            "FID Score": f"{fid_dcgan:.2f}",
            "Kestabilan Training": "Cukup Stabil (Sensitif Arsitektur)",
        },
        {
            "Arsitektur Model": "Denoising Diffusion (DDPM)",
            "Dim Laten": "28x28 (Dimensi Citra)",
            "Training Time / Epoch (s)": f"{history.get('ddpm', {}).get('avg_epoch_time', 0):.2f}",
            "Latency (ms / batch 64)": f"{lat_ddpm:.2f}",
            "Test MSE": "N/A (Denoising Model)",
            "Test PSNR (dB)": "N/A",
            "Test SSIM": "N/A",
            "FID Score": f"{fid_ddpm:.2f}",
            "Kestabilan Training": "Sangat Stabil",
        },
    ]

    df_table = pd.DataFrame(table_rows)
    print("\n=== TABEL IV: KOMPARASI METRIK NUMERIK (TEST SET) ===")
    print(df_table.to_string(index=False))
    df_table.to_csv("outputs/metrics_table.csv", index=False)

    with open("outputs/part2_mlp_comparison.json", "w") as f:
        json.dump(
            {
                "mlp_test_accuracy": history.get("mlp_test_acc", 0.0),
                "mlp_frozen_decoder": mlp_recon_metrics,
                "vanilla_ae": vae_vanilla_metrics,
            },
            f,
            indent=2,
        )

    print("\n--- Generating Mandatory Figures 1 to 5 ---")
    generate_figure_1_reconstructions(vanilla_ae, sparse_ae, vae, test_loader, device)
    generate_figure_2_random_sampling(vae, vanilla_gan, dcgan, ddpm, device)
    generate_figure_3_mlp_proof(mlp_clf, frozen_dec, vanilla_ae, test_loader, device)
    generate_figure_4_latent_interpolation(vanilla_ae, sparse_ae, vae, vanilla_gan, dcgan, test_loader, device)
    generate_figure_5_losses_and_ddpm_trajectory(history, ddpm, device)
    print("\nAll evaluations and visualizations completed successfully!")


if __name__ == "__main__":
    main()