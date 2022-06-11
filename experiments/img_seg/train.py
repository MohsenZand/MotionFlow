import sys
import os
from absl import app, flags
from pathlib import Path
from colorama import Fore, Style
from tqdm import tqdm
import numpy as np 
import torch 
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models import MotionFlowModel
import flags_img_seg 
FLAGS = flags.FLAGS

file = Path(__file__).resolve()
dir = file.parents[2]
sys.path.append(str(dir))
from utils import select_device, increment_path, count_parameters, save_model, load_state, compute_accuracy
from datasets import HorseDataset



##############################################
class Trainer():
    def __init__(self):
        super().__init__()

        print(f'{Fore.YELLOW}') 
        device = select_device(FLAGS.device, batch_size=FLAGS.batch_size)
        print(f'{Style.RESET_ALL}')

        self.data_dir = FLAGS.data_dir
        
        model = MotionFlowModel(FLAGS).to(device) 
        
        print(f'{Fore.YELLOW}') 
        print('number of param: {}'.format(count_parameters(model)))
        print(f'{Style.RESET_ALL}')
        
        params = model.parameters()
        optim = torch.optim.Adam(params, lr=FLAGS.lr, betas=FLAGS.betas, weight_decay=FLAGS.regularizer)
        
        scheduler = None
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=5, verbose=True)

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
        self.optim = optim
        self.device = device
        self.scheduler = scheduler
        self.tb_writer = SummaryWriter(log_dir=log_dir) 
        self.global_step = FLAGS.steps

    def train(self):    
        device = self.device
        data_dir = self.data_dir
        checkpoints_dir = self.checkpoints_dir

        train_data = HorseDataset(data_dir, (FLAGS.x_size[1], FLAGS.x_size[2]), FLAGS.x_size[0], "train")
        val_data = HorseDataset(data_dir, (FLAGS.x_size[1], FLAGS.x_size[2]), FLAGS.x_size[0], portion="valid")

        train_loader = DataLoader(train_data, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True)                       

        for self.epoch in range(FLAGS.init_epoch, FLAGS.num_epochs):
            self.model.train()
            
            print(f'{Fore.YELLOW}', 'Training on {}. Epoch/batch_size/device/lr: {}/{}/{}/{}'.format(
                    FLAGS.db, self.epoch, FLAGS.batch_size, device, FLAGS.lr), f'{Style.RESET_ALL}')
            
            nb = len(train_loader)
            pbar = enumerate(train_loader)
            pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}')  # progress bar

            epoch_loss = 0.0
            
            for _, batch in pbar:
                self.optim.zero_grad()

                x = batch['x'].to(device)
                y = batch['y'].to(device)

                processed_y = train_data.preprocess(y, FLAGS.label_scale, FLAGS.label_bias, FLAGS.y_bins, True)
                processed_x = train_data.preprocess(x, 1.0, 0.0, FLAGS.x_bins, True)

                _, nll = self.model(processed_x, processed_y)

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
                
            epoch_loss = float(epoch_loss / float(nb))

            if self.scheduler is not None:
                self.scheduler.step(epoch_loss)                

            print(f'{Fore.YELLOW}Validation is starting ... {Style.RESET_ALL}')
            
            self.model.eval()
            val_metric, val_loss = self.validation(val_data)
            
            self.tb_writer.add_scalar('train_loss', epoch_loss, self.epoch)
            self.tb_writer.add_scalar('val_loss', val_loss, self.epoch)
            self.tb_writer.add_scalar('mean_iou', val_metric[2], self.epoch)
            
            print(['acc', 'acc_cls', 'mean_iu (main)', 'fwavacc'])
            print(f'{Fore.GREEN}', val_metric, f'{Style.RESET_ALL}')
            
            print(f'{Fore.YELLOW}Validation finished!{Style.RESET_ALL}')
            
            save_model(self.model, self.optim, self.scheduler, checkpoints_dir, self.global_step, self.epoch)
 

    def validation(self, val_data):
        device = self.device

        valid_loader = DataLoader(val_data, batch_size=FLAGS.batch_size, shuffle=False, drop_last=False)

        val_loss = 0.0
        metrics = []

        nb = len(valid_loader)
        pbar = enumerate(valid_loader)
        pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}')  # progress bar

        with torch.no_grad():
            for bi, batch in pbar:
                x = batch['x'].to(device)
                y = batch['y'].to(device)

                sample, nll = self.model(x, y=None, reverse=True)
                loss = torch.mean(nll)
                val_loss += loss.item()

                sample = val_data.postprocess(sample, FLAGS.label_scale, FLAGS.label_bias)
                
                y_pred_imgs, y_pred_seg = val_data.convert_to_img(sample)
                y_true_imgs, y_true_seg = val_data.convert_to_img(y)

                output = None
                for i in range(0, len(y_true_imgs)):
                    true_img = y_true_imgs[i]
                    pred_img = y_pred_imgs[i]
                    row = torch.cat((x[i].cpu(), true_img, pred_img), dim=1)
                    if output is None:
                        output = row
                    else:
                        output = torch.cat((output,row), dim=2)
                save_image(output, os.path.join(self.log_dir, "trues-{}.png".format(bi)))

                acc, acc_cls, mean_iu, fwavacc = compute_accuracy(y_true_seg, y_pred_seg, 2)
                metrics.append([acc, acc_cls, mean_iu, fwavacc])              
                
                pbar.set_description('Validation/Epoch: {}'.format(self.epoch))

        val_loss = float(val_loss / float(nb))
        mean_metrics = np.mean(metrics, axis=0)

        return mean_metrics, val_loss


def main(_):
    trainer = Trainer()
    trainer.train()




if __name__ == '__main__':
    app.run(main)
