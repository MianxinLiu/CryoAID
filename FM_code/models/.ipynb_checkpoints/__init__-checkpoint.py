import torch
import numpy as np
import timm


__all__ = ['list_models', 'get_model', 'get_custom_transformer']


__implemented_models = {
    'resnet50': 'image-net',
    'ctranspath': 'models/ckpts/ctranspath.pth',
    'gigapath': 'models/ckpts/gigapath.bin',
    'uni2': 'models/ckpts/uni2.bin',
    'uni': 'models/ckpts/uni.bin',
    'chief': 'models/ckpts/chief/CHIEF_CTransPath.pth',
    'virchow2': 'huggingface download',
}


def list_models():
    print('The following are implemented models:')
    for k, v in __implemented_models.items():
        print('{}: {}'.format(k, v))
    return __implemented_models


def get_model(model_name, device, gpu_num):
    """_summary_

    Args:
        model_name (str): the name of the requried model
        device (torch.device): device, e.g. 'cuda'
        gpu_num (int): the number of GPUs used in extracting features

    Raises:
        NotImplementedError: if the model name does not exist

    Returns:
        nn.Module: model
    """
    if model_name == 'resnet50':
        from models.resnet_custom import resnet50_baseline
        model = resnet50_baseline(pretrained=True).to(device)
        
    elif model_name == 'ctranspath':
        from models.ctrans import ctranspath
        print('\n!!!! please note that ctranspath requires the modified timm 0.5.4, you can find package at here: models/ckpts/timm-0.5.4.tar , please install if needed ...\n')
        model = ctranspath(ckpt_path=__implemented_models['ctranspath']).to(device)

    elif model_name.lower() == 'uni':
        from models.uni import get_uni_model
        model = get_uni_model(device)
    
    elif model_name.lower() == 'uni2':
        from models.uni2 import get_uni2_model
        model = get_uni2_model(device)
            
    elif model_name == 'gigapath':
        from models.gigapath import get_gigapath_model
        model = get_gigapath_model(device)
                
    elif model_name == 'virchow2':
        from models.virchow2 import get_virchow_model
        model = get_virchow_model(device)
        
    elif model_name == 'chief':
        from models.chief.ctran import get_model
        model = get_model(device=device)
            
    else:
        raise NotImplementedError(f'{model_name} is not implemented')
    
    if model_name in ['resnet50']:
        if gpu_num > 1:
            model = torch.nn.parallel.DataParallel(model)
        model = model.eval()
        
    return model


def get_custom_transformer(model_name):
    """_summary_

    Args:
        model_name (str): the name of model

    Raises:
        NotImplementedError: not implementated

    Returns:
        torchvision.transformers: the transformers used to preprocess the image
    """
    if model_name in ['resnet50']:
        from models.resnet_custom import custom_transforms
        custom_trans = custom_transforms()
        
    elif model_name.lower() == 'uni':
        from models.uni import get_uni_trans
        custom_trans = get_uni_trans()
        
    elif model_name.lower() == 'uni2':
        from models.uni2 import get_uni2_trans
        custom_trans = get_uni2_trans()
    
    elif model_name == 'ctranspath':
        from models.ctrans import ctranspath_transformers
        custom_trans = ctranspath_transformers()
      
    elif model_name.lower() == 'gigapath':
        from models.gigapath import get_gigapath_trans
        custom_trans = get_gigapath_trans()
        
    elif model_name == 'virchow2':
        from models.virchow2 import get_virchow_trans
        custom_trans = get_virchow_trans()
        
    elif model_name == 'chief':
        from models.chief.ctran import get_trans
        custom_trans = get_trans()
        
    else:
        raise NotImplementedError('Transformers for {} is not implemented ...'.format(model_name))

    return custom_trans
