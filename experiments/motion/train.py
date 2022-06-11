import sys
import os
from absl import app, flags
from pathlib import Path
from colorama import Fore, Style
from tqdm import tqdm
import numpy as np 
import torch 
from torch.utils.tensorboard import SummaryWriter

from models import MotionFlowModel
from visualization import CMUForwardKinematics
from metrics import MetricsEngine
import flags_motion
FLAGS = flags.FLAGS

file = Path(__file__).resolve()
dir = file.parents[2]
sys.path.append(str(dir))
from utils import select_device, increment_path, count_parameters, save_model, load_state
from datasets import TFRecordMotionDataset


##############################################
class Trainer():
    def __init__(self):
        super().__init__()

        print(f'{Fore.YELLOW}') 
        device = select_device(FLAGS.device, batch_size=FLAGS.batch_size)
        print(f'{Style.RESET_ALL}')
        
        data_dir = os.path.join(FLAGS.data_dir, FLAGS.db)
        self.input_seq_len = FLAGS.input_seq_len
        self.target_seq_len = FLAGS.target_seq_len

        # dataset
        if FLAGS.dynamic_train_split:
            train_data_split = 'training_dynamic'
        else:
            train_data_split = 'training'

        if FLAGS.dynamic_val_split:
            val_data_split = 'validation_dynamic'
        else:
            val_data_split = 'validation'
        
        self.train_data_path = os.path.join(data_dir, 'aa', train_data_split, FLAGS.db + '-?????-of-?????')
        self.meta_data_path = os.path.join(data_dir, 'aa', 'training', 'stats.npz')
        self.val_data_path = os.path.join(data_dir, 'aa', val_data_split, FLAGS.db + '-?????-of-?????')

        model = MotionFlowModel(FLAGS).to(device) 
        
        print(f'{Fore.YELLOW}') 
        print('number of param: {}'.format(count_parameters(model)))
        print(f'{Style.RESET_ALL}')
        
        params = model.parameters()
        optim = torch.optim.Adam(params, lr=FLAGS.lr, betas=FLAGS.betas, weight_decay=FLAGS.regularizer)
        
        scheduler = None

        if FLAGS.load_weights:
            latest_checkpoint = Path(FLAGS.run_dir) / Path(FLAGS.exp_name) / 'checkpoints/last.pth.tar'
            cuda = False if device == 'cpu' else True 
            state = load_state(latest_checkpoint, cuda)
            optim.load_state_dict(state['optim'])
            model.load_state_dict(state['model'])
            FLAGS.steps = state['iteration'] + 1
            FLAGS.init_epoch = state['epoch'] + 1
            if scheduler is not None and state.get('scheduler', None) is not None:
                scheduler.load_state_dict(state['scheduler'])
            del state

        # Create metrics engine
        target_seq_len_metric = self.input_seq_len
        fk_engine = CMUForwardKinematics()
        metric_target_lengths = FLAGS.METRIC_TARGET_LENGTHS_CMU_25FPS
            
        target_lengths = [x for x in metric_target_lengths if x <= target_seq_len_metric]
        
        metrics_engine = MetricsEngine(fk_engine, target_lengths, rep='aa', which=['mpjpe'], force_valid_rot=True)
        metrics_engine.reset()

        if not FLAGS.load_weights:
            log_dir = increment_path(Path(FLAGS.run_dir) / FLAGS.exp_name, exist_ok=FLAGS.exist_ok)  
        else:
            log_dir = Path(FLAGS.run_dir) / FLAGS.exp_name

        checkpoints_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        self.model = model
        self.log_dir = log_dir
        self.checkpoints_dir = checkpoints_dir
        self.metrics_engine = metrics_engine
        self.optim = optim
        self.device = device
        self.scheduler = scheduler
        self.tb_writer = SummaryWriter(log_dir=log_dir) 
        self.target_lengths = target_lengths
        self.fk_engine = fk_engine
        self.global_step = FLAGS.steps
        
    def train(self):    
        device = self.device
        train_data_path = self.train_data_path
        val_data_path = self.val_data_path
        meta_data_path = self.meta_data_path
        checkpoints_dir = self.checkpoints_dir

        input_seq_len = self.input_seq_len
        target_seq_len = self.target_seq_len
        windows_size = (input_seq_len + target_seq_len) * 2
        inds = range(0, (input_seq_len + target_seq_len) * 2, 2)

        train_data = TFRecordMotionDataset(data_path=train_data_path, meta_data_path=meta_data_path, 
                                            batch_size=FLAGS.batch_size, shuffle=True, 
                                            windows_size=windows_size, window_type='random', 
                                            num_parallel_calls=1, normalize=FLAGS.normalize)
        
        val_data = TFRecordMotionDataset(data_path=val_data_path, meta_data_path=meta_data_path, 
                                            batch_size=FLAGS.batch_size, shuffle=True, 
                                            windows_size=input_seq_len*4, window_type='random', 
                                            num_parallel_calls=1,normalize=FLAGS.normalize)

        for self.epoch in range(FLAGS.init_epoch, FLAGS.num_epochs):
            self.model.train()
            
            print(f'{Fore.YELLOW}', 'Training on {}. Epoch/batch_size/device/lr: {}/{}/{}/{}'.format(
                    FLAGS.db, self.epoch, FLAGS.batch_size, device, FLAGS.lr), f'{Style.RESET_ALL}')
            
            train_samples = train_data.get_tf_samples()
            nb = len(train_data)
            pbar = enumerate(train_samples)
            pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}')  # progress bar

            epoch_loss = 0.0
            
            for _, batch in pbar:
                self.optim.zero_grad()
                act = batch['id']
                act = [action.decode('utf-8').split('/')[-1] for action in act]
                inputs = batch['inputs'][:, inds, :]
                ids = [action in ['walking', 'basketball'] for action in act]
                inputs = inputs[ids]

                if inputs.shape[0] <= 0:
                    continue

                y = torch.tensor(inputs).to(device)
                x = y[:, :input_seq_len]
                y = y[:, input_seq_len:]

                _, nll = self.model(x, y)

                loss = torch.mean(nll)

                # backward
                self.model.zero_grad()
                self.optim.zero_grad()
                loss.backward()
                    
                # operate grad
                if FLAGS.max_grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), FLAGS.max_grad_clip)
                if FLAGS.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), FLAGS.max_grad_norm)

                self.optim.step()
                epoch_loss += loss.item()
                pbar.set_description('Iter/Epoch:{}/{}  Loss:{:.5f}'.format(self.global_step, self.epoch, loss.data))
                self.global_step += 1
                
            if self.scheduler is not None:
                self.scheduler.step()
                
            epoch_loss = float(epoch_loss / float(nb))

            print(f'{Fore.YELLOW}Validation is starting ... {Style.RESET_ALL}')
            
            self.model.eval()
            val_loss = self.validation(val_data)

            self.metrics_engine.train_loss = epoch_loss
            self.metrics_engine.val_loss = val_loss
            final_metrics = self.metrics_engine.get_final_metrics()

            s = self.metrics_engine.get_summary_string_all(final_metrics, self.target_lengths, 
                                                        at_mode=True, tb_writer=self.tb_writer, 
                                                        step=self.epoch, training=True, 
                                                        train_loss=epoch_loss, val_loss=val_loss)
            print(f'{Fore.GREEN}', s, f'{Style.RESET_ALL}')
            
            self.metrics_engine.reset()

            print(f'{Fore.YELLOW}Validation finished!{Style.RESET_ALL}')
            
            save_model(self.model, self.optim, self.scheduler, checkpoints_dir, self.global_step, self.epoch)
 

    def validation(self, val_data):
        device = self.device
        input_seq_len = self.input_seq_len
        target_seq_len = self.target_seq_len
        inds = range(0, input_seq_len * 4, 2)
        val_samples = val_data.get_tf_samples()
        val_loss = 0.0

        nb = len(val_data)
        pbar = enumerate(val_samples)
        pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}')  # progress bar
        
        with torch.no_grad():
            for bi, batch in pbar:
                act = batch['id']
                act = [action.decode('utf-8').split('/')[-1] for action in act]
                inputs = batch['inputs'][:, inds, :]
                ids = [action in ['walking', 'basketball'] for action in act]
                inputs = inputs[ids]

                if inputs.shape[0] <= 0:
                    continue

                y = torch.tensor(inputs).to(device)
                x = y[:, :input_seq_len]
                y = y[:, input_seq_len:]

                sample, nll = self.model(x, y=None, reverse=True)
                val_loss += torch.mean(nll).data
                gt_motion = y.cpu().numpy()  
                pred_motion = sample.cpu().numpy() 
                
                pred_motion = val_data.unnormalize_zero_mean_unit_variance_channel({'poses': pred_motion}, 'poses')
                gt_motion = val_data.unnormalize_zero_mean_unit_variance_channel({'poses': gt_motion}, 'poses')
                self.metrics_engine.compute_and_aggregate(pred_motion['poses'], gt_motion['poses'])
                
                pbar.set_description('Validation/Epoch: {}'.format(self.epoch))

        mean_val_loss = float(val_loss / float(nb))

        return mean_val_loss


def main(_):
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    app.run(main)