import os
import numpy as np
import functools
import tensorflow as tf 
from colorama import Fore, Style
from torch.utils.data import Dataset
import torch
import PIL.Image as Image
from torch.utils import data
import torchvision.transforms as transforms



class TimeSeries(Dataset):
	def __init__(self, data, seq_len, train=True):
		self.seq_len = seq_len
		self.train = train
		self.frames = torch.tensor(data).T


	def __getitem__(self, idx):
		sample = self.frames[idx:idx+self.seq_len, :]
		if self.train:
			return sample
		else:
			return self.frames[-self.seq_len:, :]

	def __len__(self):
		if self.train:
			return len(self.frames) - self.seq_len
		else:
			return 1


class SmallSynthData(Dataset):
	def __init__(self, data_path, mode, params):
		self.mode = mode
		self.data_path = data_path
		if self.mode == 'train':
			path = os.path.join(data_path, 'train_feats')
			edge_path = os.path.join(data_path, 'train_edges')
		elif self.mode == 'val':
			path = os.path.join(data_path, 'val_feats')
			edge_path = os.path.join(data_path, 'val_edges')
		elif self.mode == 'test':
			path = os.path.join(data_path, 'test_feats')
			edge_path = os.path.join(data_path, 'test_edges')
		self.feats = torch.load(path)
		self.edges = torch.load(edge_path)
		self.same_norm = params['same_data_norm']
		self.no_norm = params['no_data_norm']
		if not self.no_norm:
			self._normalize_data()

	def _normalize_data(self):
		train_data = torch.load(os.path.join(self.data_path, 'train_feats'))
		if self.same_norm:
			self.feat_max = train_data.max()
			self.feat_min = train_data.min()
			self.feats = (self.feats - self.feat_min)*2/(self.feat_max-self.feat_min) - 1
		else:
			self.loc_max = train_data[:, :, :, :2].max()
			self.loc_min = train_data[:, :, :, :2].min()
			self.vel_max = train_data[:, :, :, 2:].max()
			self.vel_min = train_data[:, :, :, 2:].min()
			self.feats[:,:,:, :2] = (self.feats[:,:,:,:2]-self.loc_min)*2/(self.loc_max - self.loc_min) - 1
			self.feats[:,:,:,2:] = (self.feats[:,:,:,2:]-self.vel_min)*2/(self.vel_max-self.vel_min)-1

	def unnormalize(self, data):
		if self.no_norm:
			return data
		elif self.same_norm:
			return (data + 1) * (self.feat_max - self.feat_min) / 2. + self.feat_min
		else:
			result1 = (data[:, :, :, :2] + 1) * (self.loc_max - self.loc_min) / 2. + self.loc_min
			result2 = (data[:, :, :, 2:] + 1) * (self.vel_max - self.vel_min) / 2. + self.vel_min
			return np.concatenate([result1, result2], axis=-1)


	def __getitem__(self, idx):
		return {'inputs': self.feats[idx], 'edges':self.edges[idx]}

	def __len__(self):
		return len(self.feats)


class HorseDataset(data.Dataset):
	def __init__(self, dir, size, n_c, portion="train"):
		self.dir = dir
		self.names = self.read_names(dir, portion)
		self.n_c = n_c
		self.size = size

	def read_names(self, dir, portion):

		path = os.path.join(dir, "{}.txt".format(portion))
		names = list()
		with open(path, "r") as f:
			for line in f:
				line = line.strip()
				name = {}
				name["img"] = os.path.join(dir, os.path.join("images", line))
				name["lbl"] = os.path.join(dir, os.path.join("labels", line))
				names.append(name)
		return names

	def __len__(self):
		return len(self.names)


	def __getitem__(self, index):

		# path
		name = self.names[index]
		img_path = name["img"]
		lbl_path = name["lbl"]
		transform = transforms.Compose([transforms.Resize(self.size), transforms.ToTensor()])

		# img
		img = Image.open(img_path).convert("RGB")
		img = transform(img)

		# lbl
		lbl = Image.open(lbl_path).convert("L")
		lbl = transform(lbl)
		lbl = torch.round(lbl)
		if self.n_c > 1:
			lbl = lbl.repeat(self.n_c,1,1)

		return {"x":img, "y":lbl, "path":img_path}
		
	
	
	def preprocess(self, x, scale, bias, bins, noise=False):

		x = x / scale
		x = x - bias

		if noise == True:
			if bins == 2:
				x = x + torch.zeros_like(x).uniform_(-0.5, 0.5)
			else:
				x = x + torch.zeros_like(x).uniform_(0, 1/bins)
		return x


	def postprocess(self, x, scale, bias):

		x = x + bias
		x = x * scale
		return x


	def convert_to_img(self, y):
		import skimage.color
		import skimage.util
		import skimage.io

		C = y.size(1)

		transform = transforms.ToTensor()
		colors = np.array([[0,0,0],[255,255,255]])/255

		if C == 1:
			seg = torch.squeeze(y, dim=1).cpu().numpy()
			seg = np.nan_to_num(seg)
			seg = np.clip(np.round(seg),a_min=0, a_max=1)

		if C > 1:
			seg = torch.mean(y, dim=1, keepdim=False).cpu().numpy()
			seg = np.nan_to_num(seg)
			seg = np.clip(np.round(seg),a_min=0, a_max=1)

		B,C,H,W = y.size()
		imgs = list()
		for i in range(B):
			label_i = skimage.color.label2rgb(seg[i], colors=colors)
			label_i = skimage.util.img_as_ubyte(label_i)
			imgs.append(transform(label_i))
		return imgs, seg



class TFRecordMotionDataset(object):
	"""
	Dataset class for CMU dataset stored as TFRecord files.
	"""
	def __init__(self, data_path, meta_data_path, batch_size, shuffle, windows_size, window_type, num_parallel_calls, normalize):
		print(f'{Fore.YELLOW}')
		print('Loading motion data from {}'.format(os.path.abspath(data_path)))
		print(f'{Style.RESET_ALL}')
		# Extract a window randomly. If the sequence is shorter, ignore it.
		self.windows_size = windows_size 
		# Whether to extract windows randomly, from the beginning or the middle of the sequence.
		self.window_type = window_type
		self.num_parallel_calls = num_parallel_calls
		self.normalize = normalize
		
		self.tf_data = None
		self.data_path = data_path
		self.batch_size = batch_size
		self.shuffle = shuffle

		# Load statistics and other data summary stored in the meta-data file.
		self.meta_data = self.load_meta_data(meta_data_path)

		self.mean_all = self.meta_data['mean_all']
		self.var_all = self.meta_data['var_all']
		self.mean_channel = self.meta_data['mean_channel']
		self.var_channel = self.meta_data['var_channel']

		self.tf_data_transformations()
		self.tf_data_normalization()
		self.tf_data_to_model()

		self.tf_samples = self.tf_data.as_numpy_iterator()


	def tf_data_transformations(self):
		"""
		Loads the raw data and apply preprocessing.
		This method is also used in calculation of the dataset statistics (i.e., meta-data file).
		"""
		tf_data_opt = tf.data.Options()

		self.tf_data = tf.data.TFRecordDataset.list_files(self.data_path, seed=1234, shuffle=self.shuffle)
		self.tf_data = self.tf_data.with_options(tf_data_opt)
		self.tf_data = self.tf_data.apply(tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=self.num_parallel_calls, block_length=1, sloppy=self.shuffle))
		self.tf_data = self.tf_data.map(functools.partial(self.parse_single_tfexample_fn), num_parallel_calls=self.num_parallel_calls)
		self.tf_data = self.tf_data.prefetch(self.batch_size*10)
		if self.shuffle:
			self.tf_data = self.tf_data.shuffle(self.batch_size*10)

		if self.windows_size > 0:
			self.tf_data = self.tf_data.filter(functools.partial(self.pp_filter))
			if self.window_type == 'from_beginning':
				self.tf_data = self.tf_data.map(functools.partial(self.pp_get_windows_beginning), num_parallel_calls=self.num_parallel_calls)
			elif self.window_type == 'from_center':
				self.tf_data = self.tf_data.map(functools.partial(self.pp_get_windows_middle), num_parallel_calls=self.num_parallel_calls)
			elif self.window_type == 'random':
				self.tf_data = self.tf_data.map(functools.partial(self.pp_get_windows_random), num_parallel_calls=self.num_parallel_calls)
			else:
				raise Exception("Unknown window type.")


	def tf_data_normalization(self):
		# Applies normalization.
		if self.normalize:
			self.tf_data = self.tf_data.map(functools.partial(self.normalize_zero_mean_unit_variance_channel, key="poses"), num_parallel_calls=self.num_parallel_calls)
		else:  # Some models require the feature size.
			self.tf_data = self.tf_data.map(functools.partial(self.pp_set_feature_size), num_parallel_calls=self.num_parallel_calls)


	def tf_data_to_model(self):
		# Converts the data into the format that a model expects. Creates input, target, sequence_length, etc.
		self.tf_data = self.tf_data.map(functools.partial(self.to_model_inputs), num_parallel_calls=self.num_parallel_calls)
		self.tf_data = self.tf_data.padded_batch(self.batch_size, padded_shapes=tf.compat.v1.data.get_output_shapes(self.tf_data))
		self.tf_data = self.tf_data.prefetch(2)
		if tf.test.is_gpu_available():
			self.tf_data = self.tf_data.apply(tf.data.experimental.prefetch_to_device('/device:GPU:0'))


	def load_meta_data(self, meta_data_path):
		"""
		Loads meta-data file given the path. It is assumed to be in numpy.
		Args:
			meta_data_path:
		Returns:
			Meta-data dictionary or False if it is not found.
		"""
		if not meta_data_path or not os.path.exists(meta_data_path):
			print("Meta-data not found.")
			return False
		else:
			return np.load(meta_data_path, allow_pickle=True)['stats'].tolist()


	def pp_set_feature_size(self, sample):
		seq_len = sample["poses"].get_shape().as_list()[0]
		sample["poses"].set_shape([seq_len, self.mean_channel.shape[0]])
		return sample


	def pp_filter(self, sample):
		return tf.shape(sample["poses"])[0] >= self.windows_size


	def pp_get_windows_random(self, sample):
		start = tf.random.uniform((1, 1), minval=0, maxval=tf.shape(sample["poses"])[0]-self.windows_size+1, dtype=tf.int32)[0][0]
		end = tf.minimum(start+self.windows_size, tf.shape(sample["poses"])[0])
		sample["poses"] = sample["poses"][start:end, :]
		sample["shape"] = tf.shape(sample["poses"])
		return sample


	def pp_get_windows_beginning(self, sample):
		# Extract a window from the beginning of the sequence.
		sample["poses"] = sample["poses"][0:self.windows_size, :]
		sample["shape"] = tf.shape(sample["poses"])
		return sample


	def pp_get_windows_middle(self, sample):
		# Window is located at the center of the sequence.
		seq_len = tf.shape(sample["poses"])[0]
		start = tf.maximum((seq_len//2) - (self.windows_size//2), 0)
		end = start + self.windows_size
		sample["poses"] = sample["poses"][start:end, :]
		sample["shape"] = tf.shape(sample["poses"])
		return sample


	def to_model_inputs(self, tf_sample_dict):
		"""
		Transforms a TFRecord sample into a more general sample representation where we use global keys to represent
		the required fields by the models.
		Args:
			tf_sample_dict:
		Returns:
		"""
		model_sample = dict()
		model_sample['seq_len'] = tf_sample_dict["shape"][0]
		model_sample['inputs'] = tf_sample_dict["poses"]
		model_sample['motion_targets'] = tf_sample_dict["poses"]
		model_sample['id'] = tf_sample_dict["sample_id"]
		return model_sample


	def parse_single_tfexample_fn(self, proto):
		feature_to_type = {
			"file_id": tf.io.FixedLenFeature([], dtype=tf.string),
			"db_name": tf.io.FixedLenFeature([], dtype=tf.string),
			"shape": tf.io.FixedLenFeature([2], dtype=tf.int64),
			"poses": tf.io.VarLenFeature(dtype=tf.float32),
		}

		parsed_features = tf.io.parse_single_example(proto, feature_to_type)
		parsed_features["poses"] = tf.reshape(tf.sparse.to_dense(parsed_features["poses"]), parsed_features["shape"])

		file_id = tf.strings.substr(parsed_features["file_id"], 0, tf.strings.length(parsed_features["file_id"]))
		parsed_features["sample_id"] = tf.strings.join([parsed_features["db_name"], file_id], separator="/")

		return parsed_features


	def normalize_zero_mean_unit_variance_all(self, sample_dict, key):
		sample_dict[key] = (sample_dict[key] - self.mean_all) / self.var_all
		return sample_dict


	def normalize_zero_mean_unit_variance_channel(self, sample_dict, key):
		sample_dict[key] = (sample_dict[key] - self.mean_channel) / self.var_channel
		return sample_dict


	def unnormalize_zero_mean_unit_variance_all(self, sample_dict, key):
		if self.normalize:
			sample_dict[key] = sample_dict[key] * self.var_all + self.mean_all
		return sample_dict


	def unnormalize_zero_mean_unit_variance_channel(self, sample_dict, key):
		if self.normalize:
			sample_dict[key] = sample_dict[key] * self.var_channel + self.mean_channel
		return sample_dict


	def get_tf_samples(self):
		self.tf_samples = self.tf_data.as_numpy_iterator()
		return self.tf_samples


	def __len__(self):
		return sum(1 for _ in self.tf_data)