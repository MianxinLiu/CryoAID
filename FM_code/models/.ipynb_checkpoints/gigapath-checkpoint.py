import timm
from PIL import Image
from torchvision import transforms
import torch

# Older versions of timm have compatibility issues. Please ensure that you use a newer version by running the following command: pip install timm>=1.0.3.
# tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)

def get_gigapath_trans():
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transform
   

def get_gigapath_model(device):
    model = timm.create_model(
        "vit_giant_patch14_dinov2", 
        img_size=224,
        in_chans=3,
        patch_size=16,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        init_values=1e-05,
        mlp_ratio=5.33334,
        num_classes=0,  # 特征提取模式
        pretrained=False,  # 加载预训练权重
    )
    model.load_state_dict(torch.load('models/ckpts/gigapath.bin', map_location="cpu"), strict=True)
    model.eval()
    
    return model.to(device)


