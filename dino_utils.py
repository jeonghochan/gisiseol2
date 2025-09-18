import torch
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as transforms


#revised-0913
class DinoUpsampleHead(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int = 256, stages: int = 3):
        super().__init__()
        def blk(cin, cout):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.GroupNorm(1, cout),   # LayerNorm([C,H,W]) 대신 안전
                nn.GELU(),
            )
        layers, c = [], in_ch
        for _ in range(stages):
            layers.append(blk(c, mid_ch)); c = mid_ch
        self.stem = nn.Sequential(*layers)
        self.proj = nn.Conv2d(mid_ch, in_ch, 1)

    def forward(self, x):                 # x: [B, C, Th, Tw]
        y = self.stem(x)
        return self.proj(y)               # -> [B, C, H*, W*]

def dilate_mask(x, iterations=1):
    dilated_mask = x.unsqueeze(0).unsqueeze(0)

    for _ in range(iterations):
        dilated_mask = F.max_pool2d(dilated_mask, kernel_size=3, stride=1, padding=1)
    
    return dilated_mask.squeeze()


def pad_and_unpad(func):
    def wrapper(self, image, feature_extractor):
        width, height = image.shape[2:]
        target_width = (width + 13) // 14 * 14
        target_height = (height + 13) // 14 * 14

        pad_width_left = (target_width - width) // 2
        pad_width_right = target_width - width - pad_width_left
        pad_height_top = (target_height - height) // 2
        pad_height_bottom = target_height - height - pad_height_top

        padded_input = F.pad(image, 
                            (pad_height_top, pad_height_bottom, pad_width_left, pad_width_right), 
                            mode='replicate')

        weights = func(self, padded_input, feature_extractor).squeeze()
        rec_weights = weights[:,:,pad_width_left: pad_width_left + width, pad_height_top: pad_height_top + height]

        return rec_weights

    return wrapper


#revised-0914
class DinoFeatureExtractor:
    def __init__(self, model_name = 'dinov2_vits14'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dino_model = torch.hub.load('facebookresearch/dinov2', model_name)
        dino_model.eval()

        self.patch_size = 14 if str(model_name).endswith("14") else 16 # for case to change to 16
        
        self.dino_model = dino_model.to(device)
        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        for p in self.dino_model.parameters(): p.requires_grad_(False) # Freeze DINO

    def extract_tokens(self, image_bchw: torch.Tensor):
        """
        image_bchw: [B,3,H,W], [0,1]
        return:
          tokens: [B, Th*Tw, C]
          Th, Tw: token spatial size
          H,  W : original image size (for later cropping)
        """
        B, C, H, W = image_bchw.shape
        ps = self.patch_size
        pad_h = (ps - (H % ps)) % ps
        pad_w = (ps - (W % ps)) % ps

        # right, left only padding
        if pad_h or pad_w:
            image_bchw = F.pad(image_bchw, (0, pad_w, 0, pad_h), mode='replicate') 

        img_prep = self.preprocess(image_bchw)
        # ⚠ no_grad 쓰지 않음: 렌더(colors)로 그라디언트 흘리려면 그래프 유지
        feats = self.dino_model.forward_features(img_prep)
        tokens = feats['x_norm_patchtokens']                 # [B, Th*Tw, C]
        Th = (H + pad_h) // ps
        Tw = (W + pad_w) // ps
        return tokens, Th, Tw, H, W


    def extract(self, image):
        img_prep = self.preprocess(image)
        with torch.no_grad():
            features_dict = self.dino_model.forward_features(img_prep)
        features = features_dict['x_norm_patchtokens']
        return features

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x

class DPT_Head(nn.Module):
    def __init__(self, features):
        super(DPT_Head, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features, features // 3, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, features // 3),
            nn.GELU(), 
            nn.Conv2d(features // 3, features // 3, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 3, features // 6, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, features // 6),
            nn.GELU(),
            nn.Conv2d(features // 6, features // 6, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 6, features // 12, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, features // 12),
            #nn.ReLU(),
            #nn.Conv2d(features // 12, features // 12, kernel_size=3, stride=1, padding=1),

            #
            #Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            #nn.Conv2d(features // 12, features // 12, kernel_size=3, stride=1, padding=1),
            #nn.GroupNorm(1, features // 12),
            
            #nn.ReLU(),
            #nn.Conv2d(features // 12, features // 12, kernel_size=1, stride=1, padding=0),
            #

            nn.GELU(),
            nn.Conv2d(features // 12, 3, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.head(x)
        return x


class ConvDPT(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.head = DPT_Head(self.feature_dim)
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()

    @pad_and_unpad
    def forward(self, image, feature_extractor):
        width, height = image.shape[2:]
        features = feature_extractor.extract(image)
        features = features.reshape(-1, width//14, height//14, self.feature_dim).permute(0, 3, 1, 2)
        logits = self.head(features)
        logits = F.interpolate(
            logits, 
            size=(width, height),
            mode='bilinear',
            align_corners=False
        )
        sigma = self.softplus(logits[:, :2])
        rho = self.tanh(logits[:, 2:3])
        covariations = torch.cat([sigma, rho], dim=1)
        return covariations