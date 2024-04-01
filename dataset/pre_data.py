import argparse
import logging
import os
import pprint

import torch
import torch.backends.cudnn as cudnn
import yaml
from util.utils import init_log

from tqdm import tqdm
from ddpm.src.utils import setup_seed
from ddpm.src.feature_extractors import create_feature_extractor, collect_features
from ddpm.guided_diffusion.dist_util import dev
from ddpm.dataset.ddpm_pre_data import ddpm_pre_ACDCDataset
from ddpm.guided_diffusion.script_util import add_dict_to_argparser, model_and_diffusion_defaults

parser = argparse.ArgumentParser(description='preprocessing')
parser.add_argument('--config', default='acdc.yaml', type=str, required=False)
parser.add_argument('--labeled_id_path', default='splits/acdc/7/labeled.txt', type=str, required=False)
parser.add_argument('--unlabeled_id_path', default='splits/acdc/7/unlabeled.txt', type=str, required=False)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--ddpmf_id_path', default='splits/acdc/7/ddpm_feature.txt', type=str, required=False)
parser.add_argument('--all_train_id_path', default='splits/acdc/train_slices.txt', type=str, required=False)
parser.add_argument('--prepare_data_batch_save_dir', default='/dataset/data/slices', type=str, required=False)
add_dict_to_argparser(parser, model_and_diffusion_defaults())

def prepare_data_batch_and_save(args, all_args):
    feature_extractor = create_feature_extractor(**all_args)
    print(f"Preparing the train set for {all_args['dataset']}...")

    dataset = ddpm_pre_ACDCDataset(
        data_dir=all_args['data_root'],
        id_path=all_args['all_train_id_path'],
        image_size=all_args['crop_size'],
        num_images=all_args['training_number'])

    if 'share_noise' in all_args and all_args['share_noise']:
        rnd_gen = torch.Generator(device=dev()).manual_seed(all_args['seed'])
        noise = torch.randn(1, 1, all_args['crop_size'], all_args['crop_size'], generator=rnd_gen, device=dev())
    else:
        noise = None

    save_dir = all_args['prepare_data_batch_save_dir']  
    print(f'Total dimension {all_args["dim"][2]}')

    for img, img_name in enumerate(tqdm(dataset)):
        img = img[None].to(dev())  
        img_name = img_name
        features = feature_extractor(img, noise=noise)
        features_cpu = collect_features(all_args, features).cpu()
        X_batch_file = os.path.join(save_dir, f'{img_name}.pt')
        torch.save(features_cpu, X_batch_file)

def main():
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r", encoding='utf-8'), Loader=yaml.Loader) 
    logger = init_log('global', logging.INFO, args.log_file)
    logger.propagate = 0
    rank, world_size = 0, 1

    setup_seed(args.seed)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    prepare_data_batch_and_save(args, all_args)


if __name__ == '__main__':
    main()