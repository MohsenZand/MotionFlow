from absl import flags

# GNERAL 
flags.DEFINE_string('data_dir', '<...>', 'Processed data path')
flags.DEFINE_string('run_dir', '<...>', 'Path to save results, logs, and checkpoints') 
flags.DEFINE_string('exp_name', 'img_seg', 'experiment name')
flags.DEFINE_string('db', 'weizmann_horse', 'db name')
flags.DEFINE_bool('exist_ok', True, 'existing name ok, do not increment')
flags.DEFINE_bool('load_weights', False, 'use pretrain weights, stored in the checkpoint path in exp dir')

# MODEL
flags.DEFINE_multi_integer('x_size', [3, 32, 32], '[4, 20, 5],[3, 10, 24], model input')
flags.DEFINE_multi_integer('y_size', [32, 3, 32], '[4, 1, 5], [24, 1, 3], model output')
flags.DEFINE_integer('flow_depth', 8, 'K, flow depth at each level')
flags.DEFINE_integer('x_hidden_channels', 128, 'x hidden channels size, 128')
flags.DEFINE_integer('y_hidden_channels', 256, 'y hidden channels size, 256')
flags.DEFINE_integer('x_hidden_size', 128, 'hidden size, 64')
flags.DEFINE_boolean('learn_top', False, 'learn top in defining mean and logs in model')

# Dataset preprocess parameters
flags.DEFINE_float('label_scale', 1, '')
flags.DEFINE_float('label_bias', 0.5, '')
flags.DEFINE_float('x_bins', 256.0, '')
flags.DEFINE_float('y_bins', 2.0, '')

# MASKING 
flags.DEFINE_integer('max_dilation', 2, 'max_dilation')
flags.DEFINE_boolean('observed_idx', None, 'observed_idx')
flags.DEFINE_string('order1', 'LR', 'left-right')
flags.DEFINE_string('order2', 'TB', 'top-bottom')
flags.DEFINE_integer('nr_logistic_mix', 10, 'number of logistic mix')
flags.DEFINE_integer('input_channels', 3, '3, number of input channels')
flags.DEFINE_boolean('conv_bias', True, 'use conv bias')
flags.DEFINE_boolean('conv_mask_weight', False, 'use conv mask weight')
flags.DEFINE_integer('nr_filters', 32, '16, number of filters')
flags.DEFINE_boolean('feature_norm_op', True, 'feature_norm_op, lambda num_channels: PONO()')
flags.DEFINE_integer('num_mix', 3, 'num_mix, 3 if input_channels == 1 else 10')
flags.DEFINE_integer('kernel_size', 3, 'kernel size')
flags.DEFINE_boolean('plot_mask', False, 'plot masks if True')

# TRAIN
flags.DEFINE_string('device', '0', 'cuda device, i.e. 0 or 0,1,2,3 or cpu') 
flags.DEFINE_integer('batch_size', 32, 'batch_size')
flags.DEFINE_integer('num_epochs', 1000, 'num_epochs')
flags.DEFINE_integer('init_epoch', 0, 'init epoch is set when loading checkpoint')
flags.DEFINE_integer('steps', 0, 'steps is set when loading checkpoint')
flags.DEFINE_float('lr', 0.0002, '0.0002,learning rate')
flags.DEFINE_integer('workers', 1, 'maximum number of dataloader workers')
flags.DEFINE_multi_float('betas', [0.9, 0.9999], 'betas')
flags.DEFINE_float('regularizer', 0, '0.0005, 0.01, 4e-3, 0.0, regularizer (weight_decay)')
flags.DEFINE_float('max_grad_clip', 0.25, 'max_grad_clip')
flags.DEFINE_float('max_grad_norm', 0, 'max_grad_norm')

# EVALUATION
flags.DEFINE_integer('num_samples', 20, 'number of samples in sampled-based inference')




