import sys
import os
import numpy as np
import torch
from absl import app, flags
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from pathlib import Path
import torchvision.transforms as transforms
import PIL.Image as Image
from torchvision import transforms
import flags_img_seg 
FLAGS = flags.FLAGS

file = Path(__file__).resolve()
dir = file.parents[2]
sys.path.append(str(dir))

from models import MotionFlowModel
from datasets import HorseDataset
from utils import select_device, increment_path, load_state, compute_accuracy


def main(_):
	device = select_device(FLAGS.device, batch_size=FLAGS.batch_size)
		
	data_path = FLAGS.data_dir
	
	model = MotionFlowModel(FLAGS).to(device) 

	latest_checkpoint = Path(FLAGS.run_dir) / Path(FLAGS.exp_name) / 'checkpoints/last.pth.tar'
	cuda = False if device == 'cpu' else True 
	state = load_state(latest_checkpoint, cuda)
	model.load_state_dict(state['model'])
	
	del state

	if not FLAGS.load_weights:
		log_dir = increment_path(Path(FLAGS.run_dir) / FLAGS.exp_name, exist_ok=FLAGS.exist_ok)  
	else:
		log_dir = Path(FLAGS.run_dir) / FLAGS.exp_name

	test_dir = log_dir /'test'
	if not os.path.exists(test_dir):
		os.makedirs(test_dir)
	test_data = HorseDataset(data_path, (FLAGS.x_size[1], FLAGS.x_size[2]), FLAGS.x_size[0], "test")
	test_loader = DataLoader(test_data, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True)
	transform_im = transforms.Compose([transforms.Resize([128,128]), transforms.ToTensor()])

	model.eval()
	metrics = []

	nb = len(test_loader)
	pbar = enumerate(test_loader)
	pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}')  # progress bar

	with torch.no_grad():
		for bi, batch in pbar:
			x = batch['x'].to(device)
			y = batch['y'].to(device)
			paths = batch['path']

			sample_list = []

			for i in range(0, FLAGS.num_samples):
				y_sample,_ = model(x, y=None, reverse=True)
				_, _ = model(x, y_sample)
				sample_list.append(y_sample)

			sample = torch.stack(sample_list)
			sample = torch.mean(sample, dim=0, keepdim=False)

			sample = test_data.postprocess(sample, 1.0, 0.5)
			
			y_pred_imgs, y_pred_seg = test_data.convert_to_img(sample)
			y_true_imgs, y_true_seg = test_data.convert_to_img(y)

			output = None
			for i in range(0, len(y_true_imgs)):
				true_img = y_true_imgs[i]
				pred_img = y_pred_imgs[i]
				_, _, iou, _ = compute_accuracy(y_true_seg[i], y_pred_seg[i], 2)
				if iou > 0.5:
					img = Image.open(paths[i]).convert("RGB")
					
					img = transform_im(img)
					row = torch.cat((true_img, pred_img), dim=1)
					if output is None:
						output = row
						images = img
					else:
						output = torch.cat((output,row), dim=2)
						images = torch.cat((images,img), dim=2)
			save_image(output, os.path.join(test_dir, "trues-{}.png".format(bi)))
			save_image(images, os.path.join(test_dir, "img-{}.png".format(bi)))

			acc, acc_cls, mean_iu, fwavacc = compute_accuracy(y_true_seg, y_pred_seg, 2)
			metrics.append([acc, acc_cls, mean_iu, fwavacc])              

	mean_metrics = np.mean(metrics, axis=0)

	print(mean_metrics)


if __name__ == '__main__':
	app.run(main)