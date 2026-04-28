import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two consecutive Conv → BN → ReLU blocks. The core building block."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    """DoubleConv + MaxPool. Returns both the skip connection and the pooled output."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    """Upsample → concat skip → DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch * 2, out_ch)  # *2 because of skip concat
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Small UNet: 3 encoder stages + bottleneck + 3 decoder stages.
    Input  : (B, 2, 256, 256)  — VV + VH channels
    Output : (B, 1, 256, 256)  — raw logits (apply sigmoid for probabilities)
    """
    def __init__(self, in_channels=2, features=[32, 64, 128, 256]):
        super().__init__()

        # Encoder
        self.enc1 = EncoderBlock(in_channels, features[0])  # 2   → 32
        self.enc2 = EncoderBlock(features[0], features[1])  # 32  → 64
        self.enc3 = EncoderBlock(features[1], features[2])  # 64  → 128

        # Bottleneck
        self.bottleneck = DoubleConv(features[2], features[3])  # 128 → 256

        # Decoder
        self.dec3 = DecoderBlock(features[3], features[2])  # 256 → 128
        self.dec2 = DecoderBlock(features[2], features[1])  # 128 → 64
        self.dec1 = DecoderBlock(features[1], features[0])  # 64  → 32

        # Final 1×1 conv → binary output
        self.final = nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(self, x):
        # Encode
        x, skip1 = self.enc1(x)  # skip1: (B, 32,  256, 256)
        x, skip2 = self.enc2(x)  # skip2: (B, 64,  128, 128)
        x, skip3 = self.enc3(x)  # skip3: (B, 128,  64,  64)

        # Bottleneck
        x = self.bottleneck(x)  #        (B, 256,  32,  32)

        # Decode
        x = self.dec3(x, skip3)  #        (B, 128,  64,  64)
        x = self.dec2(x, skip2)  #        (B,  64, 128, 128)
        x = self.dec1(x, skip1)  #        (B,  32, 256, 256)

        return self.final(x)  #        (B,   1, 256, 256)


def get_model(in_channels=2, features=[32, 64, 128, 256], device="cuda"):
    model = UNet(in_channels=in_channels, features=features)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ UNet ready on {device}")
    print(f"   Parameters : {total_params:,}")
    print(f"   Input      : (B, {in_channels}, 256, 256)")
    print("   Output     : (B, 1, 256, 256) — raw logits")
    return model