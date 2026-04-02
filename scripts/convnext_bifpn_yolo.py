import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


# =========================================================
# BiFPN Layer
# =========================================================
class BiFPNLayer(nn.Module):
    def __init__(self, channels: int = 256):
        super().__init__()

        # top-down fusion weights
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        # bottom-up fusion weights
        self.w2 = nn.Parameter(torch.ones(2, dtype=torch.float32))

        self.convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            for _ in range(4)
        ])

        self.bns = nn.ModuleList([
            nn.BatchNorm2d(channels)
            for _ in range(4)
        ])

        self.act = nn.SiLU()

    @staticmethod
    def _normalize_weights(w: torch.Tensor) -> torch.Tensor:
        w = F.relu(w)
        return w / (w.sum() + 1e-6)

    def _post_fusion(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return self.act(self.bns[idx](self.convs[idx](x)))

    def forward(self, feats):
        """
        feats: [P2, P3, P4, P5]
        each feature map shape: [B, C, H, W]
        """
        if len(feats) != 4:
            raise ValueError(f"BiFPN expects 4 feature maps, got {len(feats)}")

        P2, P3, P4, P5 = feats

        w1 = self._normalize_weights(self.w1)
        w2 = self._normalize_weights(self.w2)

        # -------------------------
        # Top-down pathway
        # -------------------------
        P5_td = P5
        P4_td = w1[0] * P4 + w1[1] * F.interpolate(P5_td, size=P4.shape[-2:], mode="nearest")
        P3_td = w1[0] * P3 + w1[1] * F.interpolate(P4_td, size=P3.shape[-2:], mode="nearest")
        P2_td = w1[0] * P2 + w1[1] * F.interpolate(P3_td, size=P2.shape[-2:], mode="nearest")

        # -------------------------
        # Bottom-up pathway
        # -------------------------
        P3_out = w2[0] * P3_td + w2[1] * F.max_pool2d(P2_td, kernel_size=2, stride=2)
        P4_out = w2[0] * P4_td + w2[1] * F.max_pool2d(P3_out, kernel_size=2, stride=2)
        P5_out = w2[0] * P5_td + w2[1] * F.max_pool2d(P4_out, kernel_size=2, stride=2)

        outs = [P2_td, P3_out, P4_out, P5_out]
        outs = [self._post_fusion(x, i) for i, x in enumerate(outs)]

        return outs


# =========================================================
# Detection Head
# =========================================================
class DetectionHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )

        self.obj_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )

        self.cls_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

        self.reg_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels, 4, kernel_size=1)
        )

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        obj = self.obj_branch(x)
        cls = self.cls_branch(x)
        reg = self.reg_branch(x)
        return obj, cls, reg


# =========================================================
# Main Model
# =========================================================
class ConvNeXtBiFPNYOLO(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        bifpn_channels: int = 256,
        bifpn_layers: int = 2,
        pretrained: bool = True,
    ):
        super().__init__()

        if pretrained:
            backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        else:
            backbone = convnext_tiny(weights=None)

        self.backbone = backbone.features

        # ConvNeXt Tiny stage output channels
        backbone_channels = [96, 192, 384, 768]

        # lateral projections to BiFPN channels
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, bifpn_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(bifpn_channels),
                nn.SiLU()
            )
            for c in backbone_channels
        ])

        self.bifpn = nn.Sequential(*[
            BiFPNLayer(channels=bifpn_channels)
            for _ in range(bifpn_layers)
        ])

        self.heads = nn.ModuleList([
            DetectionHead(bifpn_channels, num_classes)
            for _ in range(4)
        ])

    def forward_backbone(self, x: torch.Tensor):
        """
        Extract 4 pyramid features from ConvNeXt Tiny.
        Uses feature outputs after blocks 1, 3, 5, 7.
        """
        feats = []

        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [1, 3, 5, 7]:
                feats.append(x)

        if len(feats) != 4:
            raise RuntimeError(f"Expected 4 backbone features, got {len(feats)}")

        return feats

    def forward(self, x: torch.Tensor):
        feats = self.forward_backbone(x)

        # lateral projection
        feats = [proj(feat) for feat, proj in zip(feats, self.laterals)]

        # BiFPN fusion
        feats = self.bifpn(feats)

        obj_out, cls_out, reg_out = [], [], []

        for feat, head in zip(feats, self.heads):
            obj, cls, reg = head(feat)
            obj_out.append(obj)
            cls_out.append(cls)
            reg_out.append(reg)

        return obj_out, cls_out, reg_out


# =========================================================
# Smoke Test
# =========================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ConvNeXtBiFPNYOLO(num_classes=5, pretrained=False).to(device)
    model.eval()

    x = torch.randn(1, 3, 640, 640).to(device)

    with torch.no_grad():
        obj_out, cls_out, reg_out = model(x)

    print("Smoke test passed.")
    for i, (o, c, r) in enumerate(zip(obj_out, cls_out, reg_out), start=2):
        print(
            f"P{i}: "
            f"obj={tuple(o.shape)}, "
            f"cls={tuple(c.shape)}, "
            f"reg={tuple(r.shape)}"
        )