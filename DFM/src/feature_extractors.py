import sys
import torch
from torch import nn
from typing import List
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_feature_extractor(model_type, **kwargs):
    """ Create the feature extractor for <model_type> architecture. """
    if model_type == 'ddpm':
        print("Creating DDPM Feature Extractor...")
        feature_extractor = FeatureExtractorDDPM(**kwargs)  
    elif model_type == 'mae':
        print("Creating MAE Feature Extractor...")
        feature_extractor = FeatureExtractorMAE(**kwargs)
    elif model_type == 'swav':
        print("Creating SwAV Feature Extractor...")
        feature_extractor = FeatureExtractorSwAV(**kwargs)
    elif model_type == 'swav_w2':
        print("Creating SwAVw2 Feature Extractor...")
        feature_extractor = FeatureExtractorSwAVw2(**kwargs)
    else:
        raise Exception(f"Wrong model type: {model_type}")
    return feature_extractor


def save_tensors(module: nn.Module, features, name: str):
    """ Process and save activations in the module. """
    if type(features) in [list, tuple]:
        features = [f.detach().float() if f is not None else None 
                    for f in features]
        setattr(module, name, features)
    elif isinstance(features, dict):
        features = {k: f.detach().float() for k, f in features.items()}
        setattr(module, name, features)
    else:
        setattr(module, name, features.detach().float())


def save_out_hook(self, inp, out):
    save_tensors(self, out, 'activations')
    return out


def save_input_hook(self, inp, out):
    save_tensors(self, inp[0], 'activations')
    return out


class FeatureExtractor(nn.Module):
    def __init__(self, model_path: str, input_activations: bool, **kwargs):
        ''' 
        Parent feature extractor class.
        
        param: model_path: path to the pretrained model
        param: input_activations: 
            If True, features are input activations of the corresponding blocks
            If False, features are output activations of the corresponding blocks
        '''
        super().__init__()
        self._load_pretrained_model(model_path, **kwargs)
        print(f"Pretrained model is successfully loaded from {model_path}")
        self.save_hook = save_input_hook if input_activations else save_out_hook
        self.feature_blocks = []

    def _load_pretrained_model(self, model_path: str, **kwargs):
        pass


class FeatureExtractorDDPM(FeatureExtractor):  
    
    def __init__(self, steps: List[int], blocks: List[int], **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        
        # Save decoder activations
        for idx, block in enumerate(self.model.output_blocks):  
            if idx in blocks:  
                block.register_forward_hook(self.save_hook)  
                self.feature_blocks.append(block)  

    def _load_pretrained_model(self, model_path, **kwargs):  
        import inspect
        import ddpm.guided_diffusion.dist_util as dist_util
        from ddpm.guided_diffusion.script_util import create_model_and_diffusion

        # Needed to pass only expected args to the function
        argnames = inspect.getfullargspec(create_model_and_diffusion)[0]
        expected_args = {name: kwargs[name] for name in argnames}
        self.model, self.diffusion = create_model_and_diffusion(**expected_args)
        
        self.model.load_state_dict(
            dist_util.load_state_dict(model_path, map_location="cpu")
        )

        self.model.to(dist_util.dev())
        if kwargs['use_fp16']:
            self.model.convert_to_fp16()
        self.model.eval()

    @torch.no_grad() 
    def forward(self, x, noise=None):  
        activations = []
        for t in self.steps:
            t = torch.tensor([t]).to(x.device)
            noisy_x = self.diffusion.q_sample(x, t, noise=noise)  
            self.model(noisy_x, self.diffusion._scale_timesteps(t)) 

            for block in self.feature_blocks:
                activations.append(block.activations)  
                block.activations = None  
     
        return activations 

def collect_features(args, activations: List[torch.Tensor], sample_idx=0):
    """ Upsample activations and concatenate them to form a feature tensor
      "dim": [256, 256, 128], "upsample_mode":"bilinear",
    """
    assert all([isinstance(acts, torch.Tensor) for acts in activations])
    size = tuple(args['dim'][:-1]) 
    dim = args['dim'][2]
    resized_activations = []
    for feats in activations:
        feats = feats[sample_idx][None]  
        feats = nn.functional.interpolate(
            feats, size=size, mode=args["upsample_mode"]  
        )
        resized_activations.append(feats[0])  
    cat_activations = torch.cat(resized_activations, dim=0)  
    cat_activations = cat_activations.unsqueeze(0).cpu()  
    conv_layer = nn.Conv2d(cat_activations.shape[1], dim, kernel_size=1)  
    aa = conv_layer(cat_activations) 
    return aa
