import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline
import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import openslide
import timm

######################################################################
import sys
sys.path.append('/ailab/user/liumianxin/transfer/')
sys.path.append('/ailab/user/liumianxin/transfer/AI_FFPE_main/')
from AI_FFPE_main.options.test_options import TestOptions
from AI_FFPE_main.models import create_model
######################################################################

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(file_path, output_path, wsi, model, FM_name, gen_model,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1, slide_file_path=''):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
		custom_downsample=custom_downsample, target_patch_size=target_patch_size)
	x, y = dataset[0]
	kwargs = {'num_workers': 0, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():	
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)

			if gen_model is not None:
				gen_model.set_input(batch)  # unpack data from data loader
				gen_model.test()           # run inference
				batch = gen_model.get_current_visuals()['fake_B']  # get image results

			if FM_name == 'pathduet': 
				features, heads = model(batch)
				features = torch.squeeze(torch.mean(features[:,2:,:],dim=1))
			elif FM_name == 'virchow2':
				output = model(batch)
				class_token = output[:, 0]    # size: 1 x 1280
				patch_tokens = output[:, 5:]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those
				# concatenate class token and average pool of patch tokens
				features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560
			else:
				features = model(batch)
				# print(features.shape)

			features = features.cpu().numpy()

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')


parser.add_argument('--data_h5_dir', type=str, default='/ailab/user/liumianxin/transfer/Cryo_path/CLAM_DATA2/')

parser.add_argument('--data_slide_dir', type=str, default='/ailab/public/pjlab-smarthealth03/liumianxin/HE/')

parser.add_argument('--slide_ext', type=str, default= '.svs')

parser.add_argument('--csv_path', type=str, default='/ailab/user/liumianxin/transfer/Cryo_path/CLAM_DATA2/process_list_autogen.csv')

parser.add_argument('--feat_dir', type=str, default='/ailab/user/liumianxin/transfer/Cryo_path/CLAM_DATA2/features_chief')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--no_gen', type=bool, default=True)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=224)
parser.add_argument('--modality', type=str)
parser.add_argument('--sub_range_l', type=int, default=0)
parser.add_argument('--sub_range_h', type=int, default=-1)
parser.add_argument('--FM', type=str, choices=['resnet', 'pathduet', 'uni', 'gigapath', 'chief', 'virchow2'], default='chief')
args = parser.parse_args()


if __name__ == '__main__':

	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)

######################################################################
	import pandas as pd
	csvdf = pd.read_csv(csv_path)
	#csvdf = csvdf[(csvdf['sdpc']==False)&(csvdf['modality']==args.modality)]
######################################################################

	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'h5_files'))
	if args.FM == 'resnet':
		# resenet50
		model = resnet50_baseline(pretrained=True)
	elif args.FM == 'pathduet':
		# PathoDuet
		sys.path.append('/ailab/user/liumianxin/transfer/PathoDuet')
		from vits import VisionTransformerMoCo
		your_checkpoint_path = '/ailab/user/liumianxin/transfer/PathoDuet/checkpoint_HE.pth'
		model = VisionTransformerMoCo(pretext_token=True, global_pool='avg')
		model.head = nn.Linear(768, 2)
		checkpoint = torch.load(your_checkpoint_path, map_location=device, weights_only=True)
		model.load_state_dict(checkpoint, strict=False)
	elif args.FM == 'uni':
		# UNI
		model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
		model.load_state_dict(torch.load('/ailab/user/liumianxin/FM_code/models/ckpts/uni.bin', map_location="cpu"), strict=True)
	elif args.FM == 'gigapath':
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
			num_classes=0,  
			pretrained=False,  
		)
		model.load_state_dict(torch.load('/ailab/user/liumianxin/FM_code/models/ckpts/gigapath.bin', map_location="cpu"), strict=True)
	elif args.FM == 'chief':
		from models.chief.ctran import get_model
		model = get_model(device=device)
	elif args.FM == 'virchow2':
		from timm.layers import SwiGLUPacked
		from timm.data import resolve_data_config
		import json
		model_path = "/ailab/user/liumianxin/FM_code/models/ckpts/virchow2/"
		config_path = f"{model_path}/config.json"
		weights_path = f"{model_path}/pytorch_model.bin"

		with open(config_path, "r") as f:
			config = json.load(f)

		model = timm.create_model(
			config["architecture"], 
			init_values=1e-5,
			num_classes=0,
			reg_tokens=4,
			mlp_ratio=5.3375,
			global_pool="",
			dynamic_img_size=True,
			pretrained=False, 
			mlp_layer=SwiGLUPacked, 
			act_layer=torch.nn.SiLU
		)

		state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
		model.load_state_dict(state_dict, strict=True)

	model.eval()
	print('loading model checkpoint')
	model = model.to(device)
	
	# print_network(model)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		
	model.eval()
	total = len(bags_dataset)
	if args.sub_range_h == -1:
		args.sub_range_h = total
	if not args.no_gen:
		# generation model
		opt = TestOptions().parse()  # get test options
		# hard-code some parameters for test
		opt.num_threads = 0   # test code only supports num_threads = 1
		opt.batch_size = 1    # test code only supports batch_size = 1
		opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
		opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
		opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
		gen_model = create_model(opt)
		gen_model.setup(opt)               # regular setup: load and print networks; create schedulers
		gen_model.parallelize()
		gen_model.eval()
	else:
		gen_model = None

	for bag_candidate_idx in range(args.sub_range_l,args.sub_range_h): 
		try:
			slide_id = bags_dataset[bag_candidate_idx].split('.')[0]
			slide_ext = bags_dataset[bag_candidate_idx].split('.')[1]
			bag_name = slide_id+'.h5'
			h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
			slide_file_path = os.path.join(args.data_slide_dir, slide_id+'.'+slide_ext)

			print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
			print(slide_id)

			if not args.no_auto_skip and slide_id+'.h5' in dest_files:
				print('skipped {}'.format(slide_id))
				continue 


			output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
			time_start = time.time()
			wsi = openslide.open_slide(slide_file_path)

			output_file_path = compute_w_loader(h5_file_path, output_path, wsi, 
			model = model, FM_name = args.FM, gen_model= gen_model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
			custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size, slide_file_path=slide_file_path)
			time_elapsed = time.time() - time_start

			print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
			file = h5py.File(output_file_path, "r")

			features = file['features'][:]
			print('features size: ', features.shape)
			print('coordinates size: ', file['coords'].shape)
			features = torch.from_numpy(features)
			bag_base, _ = os.path.splitext(bag_name)
			torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))
			del features, wsi
		except:
			print(slide_id)

		