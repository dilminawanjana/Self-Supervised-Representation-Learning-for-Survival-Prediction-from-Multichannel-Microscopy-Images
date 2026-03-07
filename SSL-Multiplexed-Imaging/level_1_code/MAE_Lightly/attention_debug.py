import types
import torch
import numpy as np
from pathlib import Path
from lightly.data import LightlyDataset

import chunked_h5_dataset
from mae_custom_transform import MAETransform

# import the EXACT training class
from main import MAE  # <-- this is your training script

def patch_torchvision_vit_store_attn(vit):
    for block in vit.encoder.layers:
        def forward_with_attn(self, x):
            y = self.ln_1(x)
            attn_out, attn_w = self.self_attention(
                y, y, y,
                need_weights=True,
                average_attn_weights=False
            )
            self.attn_weights = attn_w.detach()
            x = x + self.dropout(attn_out)
            x = x + self.mlp(self.ln_2(x))
            return x
        block.forward = types.MethodType(forward_with_attn, block)

def main():
    CKPT = "/proj/berzelius-2025-315/users/x_dilwi/SSL_method/results/mae_local_run1/epochepoch=0335.ckpt"
    H5DIR = "/proj/berzelius-2025-315/users/x_dilwi/SSL_method/patches_542_h5_lzf/patches_542_lzf"
    DEVICE = "cuda"

    # make args object similar to training
    class Args: pass
    args = Args()
    args.lr = 0.03
    args.mask_ratio = 0.75
    args.input_size = 224
    args.seed = 0
    # n_channels will be filled below

    mean_std, args.n_channels = chunked_h5_dataset.get_mean_std()
    transform = MAETransform(input_size=args.input_size, min_scale=0.2, normalize=mean_std)

    base = chunked_h5_dataset.h5_chunk_wrapper(Path(H5DIR))
    dataset = LightlyDataset.from_torch_dataset(base, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    model = MAE(args)

    # PATCH THE *REAL* VIT INSIDE THE TRAINED MODEL
    # (torchvision vit is created inside MAE.__init__)
    # so we patch AFTER init:
    vit = None
    # find vit inside the backbone (most likely place)
    if hasattr(model.backbone, "vit"):
        vit = model.backbone.vit
    elif hasattr(model, "vit"):
        vit = model.vit
    else:
        raise RuntimeError("Could not locate vit inside model/backbone")
    patch_torchvision_vit_store_attn(vit)

    ckpt = torch.load(CKPT, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=True)
    print("Loaded ckpt OK")  # with strict=True, this will error if mismatch

    model.eval().to(DEVICE)

    batch = next(iter(loader))
    images = batch[0][0].to(DEVICE)

    with torch.no_grad():
        _ = model.backbone(images=images)

    layer_idx = 11
    w = vit.encoder.layers[layer_idx].attn_weights  # [B, heads, T, T]
    print("attn stats:", w.min().item(), w.max().item(), w.mean().item())
    print("sum check:", w[0,0,0,:].sum().item())

    attn = w[0, :, 0, 1:]     # CLS -> patches
    grid = int(np.sqrt(attn.shape[-1]))
    thattn = attn.reshape(attn.shape[0], grid, grid).cpu().numpy()

    import matplotlib.pyplot as plt
    eps = 1e-8
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    for i, ax in enumerate(axes.flat):
        a = thattn[i]
        den = a.max() - a.min()
        pp = np.zeros_like(a) if den < eps else (a - a.min()) / (den + eps)
        im = ax.imshow(pp, cmap="turbo")
        ax.set_title(f"Head {i}")
        ax.axis("off")
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("attention_heads_epoch0335.png", dpi=200)
    print("Saved attention_heads_epoch0335.png")

if __name__ == "__main__":
    main()