from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests
import torch

def print_data_info(inputs):
    tensor = inputs['pixel_values']
    print('Device:{}, shape:{}, max:{:.4f}, min: {:.4f}'.format(tensor.device,
            tensor.shape, tensor.max(), tensor.min()))


if __name__ == '__main__':
    from torchvision.transforms import functional as F
    import numpy as np
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    image = torch.from_numpy(np.array(image))
    image = torch.stack([image, image], dim=0)
    pass
