import sys
import os
from absl import app, flags
from pathlib import Path
from colorama import Fore, Style
from tqdm import tqdm
import numpy as np 
import torch 

from models import MotionFlowModel
from visualization import CMUForwardKinematics, Visualizer
from metrics import MetricsEngine
import flags_motion
FLAGS = flags.FLAGS

file = Path(__file__).resolve()
dir = file.parents[2]
sys.path.append(str(dir))
from utils import select_device, load_state
from datasets import TFRecordMotionDataset



##############################################
def evaluate_model(test_data, model, metrics_engine, device):
    # make a full pass on the validation or test dataset and compute the metrics
    eval_result = dict()
    metrics_engine.reset()

    input_seq_len = FLAGS.input_seq_len
    inds = range(0, input_seq_len * 4 , 2)

    test_samples = test_data.get_tf_samples()
    nb = len(test_data)
    pbar = enumerate(test_samples)
    pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}')

    with torch.no_grad():
        for _, batch in pbar:
            act = batch['id']
            act = [action.decode('utf-8').split('/')[-1] for action in act]
            inputs = batch['inputs'][:, inds, :]
            ids = [action in ['basketball', 'walking'] for action in act]
            ids_basketball = [action in ['basketball'] for action in act]
            ids_walking = [action in ['walking'] for action in act]
            inputs = inputs[ids]
            data_id = batch['id']
            data_id = data_id[ids]

            if inputs.shape[0] <= 0:
                continue
        
            seed_sequence = inputs[:, :input_seq_len, :]
            
            y = torch.tensor(inputs).to(device)
            x = y[:, :input_seq_len]
            y = y[:, input_seq_len:]

            if FLAGS.num_samples > 1:
                # sampled_based_prediction
                sample_list = list()
                for _ in range(FLAGS.num_samples):
                    y_sample, _ = model(x, y=None, reverse=True)
                    model(x, y_sample)
                    sample_list.append(y_sample)
                sample_i = torch.stack(sample_list)    
                sample = torch.mean(sample_i, dim=0, keepdim=False)
                ###
            else:
                sample, _ = model(x, y=None, reverse=True)
            
            gt_motion = y.cpu().numpy()  
            pred_motion = sample.cpu().numpy() 
            
            pred_motion = test_data.unnormalize_zero_mean_unit_variance_channel({"poses": pred_motion}, "poses")
            gt_motion = test_data.unnormalize_zero_mean_unit_variance_channel({"poses": gt_motion}, "poses")
            seed = test_data.unnormalize_zero_mean_unit_variance_channel({"poses": seed_sequence}, "poses")
            # Store each test sample and corresponding predictions with the unique sample IDs.
            for i in range(pred_motion['poses'].shape[0]):
                seq_name = data_id[i].decode('utf-8').split('/')[-1]
                if seq_name not in eval_result.keys():
                    eval_result[seq_name] = []
                eval_result[seq_name].append((seed['poses'][i], pred_motion['poses'][i], gt_motion['poses'][i]))
            
            pbar.set_description('Processing sequence: {}'.format(seq_name))

        # finalize the computation of the metrics
        final_metrics = {}
        seq_names = eval_result.keys()
        for _, k in enumerate(seq_names):
            metrics_engine.reset()
            for idx in range(len(eval_result[k])):
                pred = eval_result[k][idx][1]
                gt = eval_result[k][idx][2]
                pred = np.expand_dims(pred, axis=0)
                gt = np.expand_dims(gt, axis=0)

                metrics_engine.compute_and_aggregate(pred, gt)
            final_metrics[k] = metrics_engine.get_final_metrics()
            
    return final_metrics, eval_result

    

##############################################
def main(_argv):
    print(f'{Fore.YELLOW}') 
    device = select_device(FLAGS.device, batch_size=FLAGS.batch_size)
    print(f'{Style.RESET_ALL}')

    data_dir = os.path.join(FLAGS.data_dir, FLAGS.db)
    eval_dir = Path(FLAGS.run_dir) / FLAGS.exp_name / 'test'
    
    if not os.path.isdir(eval_dir):
        os.mkdir(eval_dir)

    model = MotionFlowModel(FLAGS).to(device)

    latest_checkpoint = Path(FLAGS.run_dir) / FLAGS.exp_name / 'checkpoints/last.pth.tar'

    state = load_state(latest_checkpoint, cuda=False)
    model.load_state_dict(state["model"])
    del state

    if FLAGS.dynamic_test_split:
        data_split = "test_dynamic"
    else:
        data_split = "test"
    
    test_data_path = os.path.join(data_dir, 'aa', data_split, FLAGS.db + '-?????-of-?????')
    meta_data_path = os.path.join(data_dir, 'aa', "training", "stats.npz")
        
    print("Loading test data from " + test_data_path)

    # Create dataset.
    test_data = TFRecordMotionDataset(data_path=test_data_path, meta_data_path=meta_data_path,
                                        batch_size=FLAGS.batch_size, shuffle=False,
                                        windows_size=FLAGS.input_seq_len*4, window_type='from_beginning',
                                        num_parallel_calls=1, normalize=FLAGS.normalize)

    print("Evaluating Model " + str(latest_checkpoint))
    
    # Create metrics engine
    target_seq_len_metric = FLAGS.input_seq_len

    fk_engine = CMUForwardKinematics()
    metric_target_lengths = FLAGS.METRIC_TARGET_LENGTHS_CMU_25FPS
    target_lengths = [x for x in metric_target_lengths if x <= target_seq_len_metric]
    metrics_engine = MetricsEngine(fk_engine, target_lengths, rep='aa', which=['mpjpe'], force_valid_rot=True)
    metrics_engine.reset()

    print("Evaluating test set...")
    
    final_metrics, eval_result = evaluate_model(test_data, model, metrics_engine, device)
    
    seq_names = final_metrics.keys()
    for _, k in enumerate(seq_names):
        test_metric = final_metrics[k]
        s = metrics_engine.get_summary_string_all(test_metric, target_lengths, at_mode=True, 
                                            tb_writer=None, step=0, training=False, 
                                            train_loss=None, val_loss=None)
        
        print('********', k, '********', s)
        print()

    if FLAGS.visualize:
        visualizer = Visualizer(interactive=FLAGS.interactive, fk_engine=fk_engine, rep='aa',
                                output_dir=eval_dir, to_video=FLAGS.to_video, save_figs=FLAGS.save_figs)

        print(f'{Fore.YELLOW}Visualizing some samples...{Style.RESET_ALL}')

        seq_names = eval_result.keys()
        for _, k in enumerate(seq_names):
            randint = torch.randint(1, len(eval_result[k]), (10,))
            for idx in randint:
                visualizer.visualize_results(
                    eval_result[k][idx][0], eval_result[k][idx][1], eval_result[k][idx][2], title=k+'_'+str(idx))



if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass