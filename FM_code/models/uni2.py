# https://huggingface.co/MahmoodLab/UNI2-h
import timm
from torchvision import transforms
import torch
    
    
def get_uni2_trans():
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transform



def get_uni2_model(device):
    timm_kwargs = {
            'model_name': 'vit_giant_patch14_224',
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
    model = timm.create_model(
        pretrained=False, **timm_kwargs
    )
    model.load_state_dict(torch.load('models/ckpts/uni2.bin', map_location="cpu"), strict=True)
    model.eval()
    
    return model.to(device)