import argparse
import json
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.seed import set_seed
from utils.dataset import get_fashion_mnist_loaders
from models.mlp_classifier import MLPClassifier, FrozenLatentDecoder
from models.autoencoders import VanillaAutoencoder, SparseAutoencoder, VariationalAutoencoder
from models.gans import VanillaGAN, DCGAN
from models.ddpm import DDPM


def train_mlp_and_frozen_decoder(train_loader, test_loader, device, epochs=15, lr=1e-3):
    print("\n=== [1/7] Training MLP Classifier & Stage 4 Frozen Decoder ===")
    mlp = MLPClassifier().to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        mlp.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = mlp(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        mlp.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = mlp(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = 100.0 * correct / total
        if epoch % 5 == 0 or epoch == epochs:
            print(f"  MLP Epoch [{epoch}/{epochs}] | Loss: {total_loss/len(train_loader):.4f} | Test Acc: {acc:.2f}%")

    torch.save(mlp.state_dict(), "outputs/checkpoints/mlp_classifier.pth")

    for param in mlp.parameters():
        param.requires_grad = False
    mlp.eval()

    frozen_decoder = FrozenLatentDecoder().to(device)
    dec_optimizer = torch.optim.Adam(frozen_decoder.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        frozen_decoder.train()
        total_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            with torch.no_grad():
                z_mlp = mlp.extract_latent(x)
            dec_optimizer.zero_grad()
            recon = frozen_decoder(z_mlp)
            loss = F.mse_loss(recon, x)
            loss.backward()
            dec_optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Frozen Decoder Epoch [{epoch}/{epochs}] | MSE Loss: {total_loss/len(train_loader):.4f}")

    torch.save(frozen_decoder.state_dict(), "outputs/checkpoints/mlp_frozen_decoder.pth")
    return acc


def train_autoencoder_family(model_name, model, train_loader, device, epochs=25, lr=1e-3):
    print(f"\n=== Training {model_name} ===")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    epoch_times = []

    for epoch in range(1, epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_t = time.perf_counter()

        total_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()

            if model_name == "vanilla_ae":
                recon, _ = model(x)
                loss = model.loss_function(recon, x)
            elif model_name == "sparse_ae":
                recon, h = model(x)
                loss = model.loss_function(recon, x, h)
            elif model_name == "vae":
                recon, mu, logvar = model(x)
                loss = model.loss_function(recon, x, mu, logvar)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_t
        epoch_times.append(elapsed)

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch [{epoch}/{epochs}] | Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")

    torch.save(model.state_dict(), f"outputs/checkpoints/{model_name}.pth")
    return {
        "loss": losses,
        "avg_epoch_time": sum(epoch_times) / len(epoch_times),
    }


def train_gan_family(model_name, model, train_loader, device, epochs=30, lr=2e-4):
    print(f"\n=== Training {model_name} ===")
    model = model.to(device)
    opt_g = torch.optim.Adam(model.generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    g_losses, d_losses = [], []
    epoch_times = []

    for epoch in range(1, epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_t = time.perf_counter()

        total_g, total_d = 0.0, 0.0
        for real_imgs, _ in train_loader:
            real_imgs = real_imgs.to(device)
            bs = real_imgs.size(0)

            real_labels = torch.full((bs, 1), 0.9, device=device)
            fake_labels = torch.zeros((bs, 1), device=device)

            opt_d.zero_grad()
            out_real = model.discriminator(real_imgs)
            loss_d_real = criterion(out_real, real_labels)

            z = torch.randn(bs, model.latent_dim, device=device)
            fake_imgs = model.generator(z)
            out_fake = model.discriminator(fake_imgs.detach())
            loss_d_fake = criterion(out_fake, fake_labels)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()

            opt_g.zero_grad()
            out_fake_for_g = model.discriminator(fake_imgs)
            loss_g = criterion(out_fake_for_g, torch.ones((bs, 1), device=device))
            loss_g.backward()
            opt_g.step()

            total_d += loss_d.item()
            total_g += loss_g.item()

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_t
        epoch_times.append(elapsed)

        avg_g = total_g / len(train_loader)
        avg_d = total_d / len(train_loader)
        g_losses.append(avg_g)
        d_losses.append(avg_d)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch [{epoch}/{epochs}] | D Loss: {avg_d:.4f} | G Loss: {avg_g:.4f} | Time: {elapsed:.2f}s")

    torch.save(model.state_dict(), f"outputs/checkpoints/{model_name}.pth")
    return {
        "g_loss": g_losses,
        "d_loss": d_losses,
        "avg_epoch_time": sum(epoch_times) / len(epoch_times),
    }


def train_ddpm(train_loader, device, epochs=25, lr=2e-4, timesteps=300):
    print(f"\n=== Training DDPM (T={timesteps}) ===")
    model = DDPM(timesteps=timesteps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    losses = []
    epoch_times = []

    for epoch in range(1, epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_t = time.perf_counter()

        total_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss_function(x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_t
        epoch_times.append(elapsed)

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch [{epoch}/{epochs}] | Noise MSE Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")

    torch.save(model.state_dict(), "outputs/checkpoints/ddpm.pth")
    return {
        "loss": losses,
        "avg_epoch_time": sum(epoch_times) / len(epoch_times),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "mlp", "vanilla_ae", "sparse_ae", "vae", "vanilla_gan", "dcgan", "ddpm"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs("outputs/checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running training on device: {device} | Seed: {args.seed}")

    train_01, _, test_01 = get_fashion_mnist_loaders(batch_size=args.batch_size, seed=args.seed, normalize_to_neg_one=False)
    train_neg1, _, _ = get_fashion_mnist_loaders(batch_size=args.batch_size, seed=args.seed, normalize_to_neg_one=True)

    history_path = "outputs/training_history.json"
    history = json.load(open(history_path)) if os.path.exists(history_path) else {}

    if args.model in ["all", "mlp"]:
        mlp_acc = train_mlp_and_frozen_decoder(train_01, test_01, device, epochs=15)
        history["mlp_test_acc"] = mlp_acc

    if args.model in ["all", "vanilla_ae"]:
        history["vanilla_ae"] = train_autoencoder_family(
            "vanilla_ae", VanillaAutoencoder(latent_dim=64), train_01, device, epochs=args.epochs
        )

    if args.model in ["all", "sparse_ae"]:
        history["sparse_ae"] = train_autoencoder_family(
            "sparse_ae", SparseAutoencoder(latent_dim=256, sparsity_lambda=1e-4), train_01, device, epochs=args.epochs
        )

    if args.model in ["all", "vae"]:
        history["vae"] = train_autoencoder_family(
            "vae", VariationalAutoencoder(latent_dim=64, beta=1.0), train_01, device, epochs=args.epochs
        )

    if args.model in ["all", "vanilla_gan"]:
        history["vanilla_gan"] = train_gan_family(
            "vanilla_gan", VanillaGAN(latent_dim=64), train_neg1, device, epochs=args.epochs
        )

    if args.model in ["all", "dcgan"]:
        history["dcgan"] = train_gan_family(
            "dcgan", DCGAN(latent_dim=100), train_neg1, device, epochs=args.epochs
        )

    if args.model in ["all", "ddpm"]:
        history["ddpm"] = train_ddpm(
            train_neg1, device, epochs=args.epochs, timesteps=300
        )

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete! History saved to {history_path}")


if __name__ == "__main__":
    main()