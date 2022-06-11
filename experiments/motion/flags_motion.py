from absl import flags

# GNERAL 
flags.DEFINE_string('data_dir', <PATH TO DATA DIR>, 'Processed data path')
flags.DEFINE_string('run_dir', <PATH TO LOG DIR>, 'Path to save results, logs, and checkpoints')  ###
flags.DEFINE_string('exp_name', 'motion', 'exp name')
flags.DEFINE_string('db', 'cmu', 'db name')
flags.DEFINE_bool('exist_ok', True, 'existing name ok, do not increment')
flags.DEFINE_bool('load_weights', True, 'use pretrain weights, stored in the checkpoint path in exp dir')

# MODEL
flags.DEFINE_multi_integer('x_size', [3, 25, 24], 'model input')
flags.DEFINE_multi_integer('y_size', [24, 1, 3], 'model output')
flags.DEFINE_integer("input_seq_len", 25, 'number of frames as a seed sequence')
flags.DEFINE_integer("target_seq_len", 3, 'number of frames that model must predict in training')
flags.DEFINE_integer('flow_depth', 8, 'K, flow depth at each level')
flags.DEFINE_integer('x_hidden_channels', 128, 'x hidden channels size')
flags.DEFINE_integer('y_hidden_channels', 256, 'y hidden channels size')
flags.DEFINE_integer('x_hidden_size', 128, 'hidden size')
flags.DEFINE_boolean('learn_top', False, 'learn top in defining mean and logs in model')
flags.DEFINE_integer('pred_length', 25, 'prediction length')
flags.DEFINE_boolean('residual', True, 'learn residuals')

# MASKING 
flags.DEFINE_integer('max_dilation', 2, 'max_dilation')
flags.DEFINE_boolean('observed_idx', None, 'observed_idx')
flags.DEFINE_string('order1', 'LR', 'left-right')
flags.DEFINE_string('order2', 'TB', 'top-bottom')
flags.DEFINE_integer('nr_logistic_mix', 10, 'number of logistic mix')
flags.DEFINE_integer('input_channels', 3, 'number of input channels')
flags.DEFINE_boolean('conv_bias', True, 'use conv bias')
flags.DEFINE_boolean('conv_mask_weight', False, 'use conv mask weight')
flags.DEFINE_integer('nr_filters', 32, 'number of filters')
flags.DEFINE_boolean('feature_norm_op', True, 'feature_norm_op, lambda num_channels: PONO()')
flags.DEFINE_integer('num_mix', 3, 'num_mix, 3 if input_channels == 1 else 10')
flags.DEFINE_integer('kernel_size', 3, 'kernel size')
flags.DEFINE_boolean('plot_mask', False, 'plot masks if True')

# TRAIN
flags.DEFINE_string('device', '0', 'cuda device, i.e. 0 or 0,1,2,3 or cpu')  ###
flags.DEFINE_integer('batch_size', 256, 'batch_size')
flags.DEFINE_integer('num_epochs', 1000, 'num_epochs')
flags.DEFINE_integer('init_epoch', 0, 'init epoch is set when loading checkpoint')
flags.DEFINE_integer('steps', 0, 'steps is set when loading checkpoint')
flags.DEFINE_float('lr', 0.0002, 'learning rate')
flags.DEFINE_integer('workers', 1, 'maximum number of dataloader workers')
flags.DEFINE_multi_float('betas', [0.9, 0.9999], 'betas')
flags.DEFINE_float('regularizer', 0, 'regularizer (weight_decay)')
flags.DEFINE_float('max_grad_clip', 0.25, 'max_grad_clip')
flags.DEFINE_float('max_grad_norm', 0, 'max_grad_norm')
flags.DEFINE_boolean("normalize", False, 'If set, zero-mean unit-variance normalization is used on data')
flags.DEFINE_boolean("dynamic_train_split", True, 'train samples are extracted on-the-fly. multiple seq samples are generated based on the window and stride sizes in preprocess_datasets.py')
flags.DEFINE_boolean("dynamic_val_split", False, 'validation samples are extracted on-the-fly. multiple seq samples are generated based on the window and stride sizes in preprocess_datasets.py')

# EVALUATION
flags.DEFINE_integer('num_samples', 1, 'number of samples in sampled-based inference')
flags.DEFINE_boolean("dynamic_test_split", True, 'test samples are extracted on-the-fly. multiple seq samples are generated based on the window and stride sizes in preprocess_datasets.py')
flags.DEFINE_boolean("visualize", True, 'Visualize ground-truth and predictions side-by-side by using human skeleton.')
flags.DEFINE_boolean("interactive", True, 'if motion is to be shown in an interactive matplotlib window.')
flags.DEFINE_boolean("to_video", False, 'Save the model predictions to mp4 videos in the experiments folder.')
flags.DEFINE_boolean("save_figs", True, 'Save the model predictions to jpeg images in the experiments folder.')

# Metrics and summaries 
flags.DEFINE_multi_integer("METRIC_TARGET_LENGTHS_CMU_25FPS", [1, 2, 4, 8, 10, 14, 25],'# @ 25 Hz, in ms: [80, 160, 320, 400, 560, 1000]')    




