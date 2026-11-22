import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from utils.dataset import FASHION_MNIST_CLASSES
from utils.metrics import denormalize_to_zero_one
from models.mlp_classifier import sample_mlp_prior


plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 9


@torch.no_grad()
def generate_figure_1_reconstructions(
    vanilla_ae, sparse_ae, vae, test_loader, device, save_path="outputs/figures/fig1_reconstructions.png"
):
    vanilla_ae.eval()
    sparse_ae.eval()
    vae.eval()

    x_batch, y_batch = next(iter(test_loader))
    imgs = x_batch[:8].to(device)
    labels = y_batch[:8].tolist()

    recon_vae_v, _ = vanilla_ae(imgs)
    recon_sae, _ = sparse_ae(imgs)
    recon_vae, _, _ = vae(imgs)

    rows = [
        ("Ground Truth", imgs.cpu()),
        ("Vanilla AE (z=64)", recon_vae_v.cpu()),
        ("Sparse AE (h=256)", recon_sae.cpu()),
        ("VAE (z=64)", recon_vae.cpu()),
    ]

    fig, axes = plt.subplots(4, 8, figsize=(12, 6.2))
    for r_idx, (row_title, tensor_imgs) in enumerate(rows):
        for c_idx in range(8):
            ax = axes[r_idx, c_idx]
            ax.imshow(tensor_imgs[c_idx, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r_idx == 0:
                ax.set_title(FASHION_MNIST_CLASSES[labels[c_idx]], fontsize=8)
            if c_idx == 0:
                ax.set_ylabel(row_title, fontsize=8.5, fontweight="bold")

    plt.suptitle("Figur 1: Perbandingan Rekonstruksi Citra Test Set (Ground Truth vs AE Family)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


@torch.no_grad()
def generate_figure_2_random_sampling(
    vae, vanilla_gan, dcgan, ddpm, device, save_path="outputs/figures/fig2_random_sampling.png"
):
    samples_dict = {
        "Panel A: VAE (z ~ N(0, I))": vae.sample(16, device).cpu(),
        "Panel B: Vanilla GAN (MLP)": denormalize_to_zero_one(vanilla_gan.sample(16, device), True).cpu(),
        "Panel C: DCGAN (Conv)": denormalize_to_zero_one(dcgan.sample(16, device), True).cpu(),
        "Panel D: DDPM (T=300)": denormalize_to_zero_one(ddpm.sample(16, device), True).cpu(),
    }

    fig = plt.figure(figsize=(11, 11))
    outer_grid = fig.add_gridspec(2, 2, wspace=0.15, hspace=0.22)

    for idx, (panel_title, imgs) in enumerate(samples_dict.items()):
        inner_grid = outer_grid[idx // 2, idx % 2].subgridspec(4, 4, wspace=0.05, hspace=0.05)
        ax_title = fig.add_subplot(outer_grid[idx // 2, idx % 2])
        ax_title.set_title(panel_title, fontsize=10.5, fontweight="bold", pad=10)
        ax_title.axis("off")

        for i in range(16):
            ax = fig.add_subplot(inner_grid[i // 4, i % 4])
            ax.imshow(imgs[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")

    plt.suptitle("Figur 2: Sintesis Sampel Acak dari Prior Gaussian N(0, I) Antarmodel Generatif", fontsize=12, fontweight="bold", y=0.98)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


@torch.no_grad()
def generate_figure_3_mlp_proof(
    mlp_clf, frozen_decoder, vanilla_ae, test_loader, device, save_path="outputs/figures/fig3_mlp_proof.png"
):
    mlp_clf.eval()
    frozen_decoder.eval()
    vanilla_ae.eval()

    x_batch, y_batch = next(iter(test_loader))
    imgs = x_batch[:6].to(device)
    labels = y_batch[:6].to(device)

    z_mlp = mlp_clf.extract_latent(imgs)
    y_onehot = F.one_hot(labels, num_classes=10).float()

    proj_class = mlp_clf.project_from_class_onehot(y_onehot).cpu()
    proj_latent = mlp_clf.project_from_latent(z_mlp).cpu()
    frozen_recon = frozen_decoder(z_mlp).cpu()
    ae_recon, _ = vanilla_ae(imgs)
    ae_recon = ae_recon.cpu()
    mlp_prior_samples = sample_mlp_prior(mlp_clf, frozen_decoder, test_loader, num_samples=6, device=device).cpu()

    rows = [
        ("1. Ground Truth", imgs.cpu()),
        ("2a. W^T from Class (y_c)", proj_class),
        ("2b. W^T from Latent (z_MLP)", proj_latent),
        ("3a. Frozen Dec (z_MLP)", frozen_recon),
        ("3b. Vanilla AE (z_AE=64)", ae_recon),
        ("4. MLP Prior Sampling", mlp_prior_samples),
    ]

    fig, axes = plt.subplots(6, 6, figsize=(9.5, 9.5))
    for r_idx, (title, tensor_imgs) in enumerate(rows):
        for c_idx in range(6):
            ax = axes[r_idx, c_idx]
            ax.imshow(tensor_imgs[c_idx, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r_idx == 0:
                ax.set_title(FASHION_MNIST_CLASSES[labels[c_idx].item()], fontsize=8)
            if c_idx == 0:
                ax.set_ylabel(title, fontsize=8, fontweight="bold")

    plt.suptitle("Figur 3: Pembuktian Empiris Batasan Bobot W^T dan Ruang Laten MLP Classifier", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


@torch.no_grad()
def generate_figure_4_latent_interpolation(
    vanilla_ae, sparse_ae, vae, vanilla_gan, dcgan, test_loader, device,
    save_path="outputs/figures/fig4_latent_interpolation.png"
):
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    x_A, x_B = None, None
    for x_b, y_b in test_loader:
        for i in range(len(y_b)):
            if y_b[i].item() == 0 and x_A is None:
                x_A = x_b[i : i + 1].to(device)
            elif y_b[i].item() == 9 and x_B is None:
                x_B = x_b[i : i + 1].to(device)
        if x_A is not None and x_B is not None:
            break

    zA_vae_v, zB_vae_v = vanilla_ae.encode(x_A), vanilla_ae.encode(x_B)
    zA_sae, zB_sae = sparse_ae.encode(x_A), sparse_ae.encode(x_B)
    zA_vae, _ = vae.encode(x_A)
    zB_vae, _ = vae.encode(x_B)

    zA_vgan = torch.randn(1, vanilla_gan.latent_dim, device=device)
    zB_vgan = torch.randn(1, vanilla_gan.latent_dim, device=device)
    zA_dcgan = torch.randn(1, dcgan.latent_dim, device=device)
    zB_dcgan = torch.randn(1, dcgan.latent_dim, device=device)

    model_specs = [
        ("Vanilla AE", lambda z: vanilla_ae.decode(z), zA_vae_v, zB_vae_v, False),
        ("Sparse AE", lambda z: sparse_ae.decode(z), zA_sae, zB_sae, False),
        ("VAE", lambda z: vae.decode(z), zA_vae, zB_vae, False),
        ("Vanilla GAN", lambda z: vanilla_gan.generator(z), zA_vgan, zB_vgan, True),
        ("DCGAN", lambda z: dcgan.generator(z), zA_dcgan, zB_dcgan, True),
    ]

    fig, axes = plt.subplots(5, len(alphas), figsize=(10.5, 8.2))
    for r_idx, (name, decode_fn, zA, zB, is_neg_one) in enumerate(model_specs):
        for c_idx, alpha in enumerate(alphas):
            z_interp = (1.0 - alpha) * zA + alpha * zB
            img = decode_fn(z_interp)
            img = denormalize_to_zero_one(img, from_neg_one=is_neg_one).cpu()

            ax = axes[r_idx, c_idx]
            ax.imshow(img[0, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r_idx == 0:
                ax.set_title(f"α = {alpha:.1f}", fontsize=9, fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel(name, fontsize=9, fontweight="bold")

    plt.suptitle("Figur 4: Interpolasi Linier Ruang Laten z(α) = (1-α)z_A + αz_B Antarmodel", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


@torch.no_grad()
def generate_figure_5_losses_and_ddpm_trajectory(
    history: dict, ddpm, device, save_path="outputs/figures/fig5_training_and_ddpm_trajectory.png"
):
    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(3, 3, hspace=0.38, wspace=0.28)

    ae_configs = [
        ("vanilla_ae", "Vanilla AE (MSE Loss)", "tab:blue"),
        ("sparse_ae", "Sparse AE (MSE + L1 Loss)", "tab:cyan"),
        ("vae", "VAE (ELBO: BCE + KLD)", "tab:green"),
    ]
    for idx, (key, title, color) in enumerate(ae_configs):
        ax = fig.add_subplot(gs[0, idx])
        if key in history:
            vals = history[key]["loss"]
            ax.plot(range(1, len(vals) + 1), vals, color=color, linewidth=1.8)
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Loss", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    gan_configs = [
        ("vanilla_gan", "Vanilla GAN (G vs D Loss)"),
        ("dcgan", "DCGAN (G vs D Loss)"),
    ]
    for idx, (key, title) in enumerate(gan_configs):
        ax = fig.add_subplot(gs[1, idx])
        if key in history:
            g_vals = history[key]["g_loss"]
            d_vals = history[key]["d_loss"]
            epochs = range(1, len(g_vals) + 1)
            ax.plot(epochs, d_vals, label="Discriminator (D)", color="tab:red", linewidth=1.6)
            ax.plot(epochs, g_vals, label="Generator (G)", color="tab:orange", linewidth=1.6)
            ax.legend(fontsize=7.5)
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("BCE Loss", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    ax_ddpm = fig.add_subplot(gs[1, 2])
    if "ddpm" in history:
        vals = history["ddpm"]["loss"]
        ax_ddpm.plot(range(1, len(vals) + 1), vals, color="tab:purple", linewidth=1.8)
    ax_ddpm.set_title("DDPM (Noise Prediction MSE)", fontsize=9.5, fontweight="bold")
    ax_ddpm.set_xlabel("Epoch", fontsize=8)
    ax_ddpm.set_ylabel("MSE Loss", fontsize=8)
    ax_ddpm.grid(True, linestyle="--", alpha=0.4)

    _, trajectory = ddpm.sample(num_samples=1, device=device, return_trajectory=True, num_steps_to_record=9)
    traj_grid = gs[2, :].subgridspec(1, len(trajectory), wspace=0.08)

    ax_traj_header = fig.add_subplot(gs[2, :])
    ax_traj_header.set_title("DDPM Reverse Denoising Trajectory (x_T → x_0)", fontsize=10, fontweight="bold", pad=12)
    ax_traj_header.axis("off")

    for i, (t_step, img_t) in enumerate(trajectory):
        ax = fig.add_subplot(traj_grid[0, i])
        img_vis = denormalize_to_zero_one(img_t, from_neg_one=True)
        ax.imshow(img_vis[0, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"t = {t_step}", fontsize=8)
        ax.axis("off")

    plt.suptitle("Figur 5: Dinamika Kurva Training Loss Keenam Model & Trajektori Denoising DDPM", fontsize=11.5, fontweight="bold", y=0.98)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")