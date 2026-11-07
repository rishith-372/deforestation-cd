"""
model.py — Siamese U-Net for before/after deforestation change detection.

Architecture (compact, self-contained, no pretrained backbone):
  * ONE shared encoder (same weights) runs over the before image and the after
    image separately — this is the "Siamese" part.
  * At every scale we take the ABSOLUTE DIFFERENCE of the two feature maps, so
    the decoder sees "what changed" rather than either date alone.
  * A U-Net decoder upsamples the differenced features (with skip connections)
    to a per-pixel loss logit.

Input : before, after  each (B, C=4, 256, 256)  scaled reflectance
Output: logits         (B, 1, 256, 256)         apply sigmoid for loss prob

Device-agnostic: runs on CPU (sanity check) or CUDA (Colab) unchanged.
"""

import torch
import torch.nn as nn


def conv_block(c_in, c_out):
    """Two 3x3 conv -> BN -> ReLU."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    """Shared encoder: returns multi-scale feature maps (for skip connections)."""

    def __init__(self, in_ch, chs):
        super().__init__()
        self.stem = conv_block(in_ch, chs[0])
        self.downs = nn.ModuleList([conv_block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        feats = [self.stem(x)]                 # full resolution
        for down in self.downs:
            feats.append(down(self.pool(feats[-1])))
        return feats                           # [256,128,64,32,16] spatial for 256 input


class SiameseUNet(nn.Module):
    def __init__(self, in_ch=4, chs=(32, 64, 128, 256, 512), n_classes=1):
        super().__init__()
        self.encoder = Encoder(in_ch, chs)     # shared across before & after

        rev = list(reversed(chs))              # [512,256,128,64,32]
        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        for i in range(len(rev) - 1):
            self.ups.append(nn.ConvTranspose2d(rev[i], rev[i + 1], 2, stride=2))
            self.decs.append(conv_block(rev[i + 1] * 2, rev[i + 1]))   # upsampled + skip
        self.head = nn.Conv2d(chs[0], n_classes, 1)

    def forward(self, before, after):
        fb = self.encoder(before)
        fa = self.encoder(after)
        diffs = [torch.abs(b - a) for b, a in zip(fb, fa)]   # change features per scale

        x = diffs[-1]                                        # bottleneck
        skips = list(reversed(diffs[:-1]))                   # finer scales, deep -> shallow
        for up, dec, skip in zip(self.ups, self.decs, skips):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        return self.head(x)                                  # (B, 1, H, W) logits


if __name__ == "__main__":
    # quick shape check (CPU)
    m = SiameseUNet()
    b = torch.randn(2, 4, 256, 256)
    a = torch.randn(2, 4, 256, 256)
    out = m(b, a)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"output {tuple(out.shape)}  params {n_params/1e6:.2f}M")
