import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from pathlib import Path

file = Path(__file__).resolve()
dir = file.parents[2]
sys.path.append(str(dir))
from masking import locally_masked_conv2d, PONO, concat_elu, mask_param, pono


################################
class MotionFlowModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.learn_top = args.learn_top
        self.n_bins = 2
        self.seq_size_x = args.x_size  
        self.seq_size_y = args.y_size  
        self.residual = args.residual
        self.pred_length = args.pred_length

        self.flow = MotionFlow(args)
        self.nn_theta = NNTheta(encoder_ch_in=self.seq_size_y[0]*2, encoder_mode='conv_net', h_ch_in=self.seq_size_y[0], num_blocks=5) 
        self.register_parameter("new_mean", nn.Parameter(torch.zeros([1, self.flow.output_shapes[-1][1], self.flow.output_shapes[-1][2], self.flow.output_shapes[-1][3]])))
        self.register_parameter("new_logs", nn.Parameter(torch.zeros([1, self.flow.output_shapes[-1][1], self.flow.output_shapes[-1][2], self.flow.output_shapes[-1][3]])))

    def forward(self, x, y=None, eps_std=1.0, reverse=False, device=None):
        B = x.size(0)
        s1, s2, s3 = self.seq_size_x
        s1y, s2y, s3y = self.seq_size_y

        dimensions = s1y*s2y*s3y
        logdet = torch.zeros(B).to(x.device)
        logdet += float(-np.log(self.n_bins) * dimensions)
        mean, logs = self.prior()   

        x = x.view(B, s2, s3, s1)
        x = x.permute(0, 3, 1, 2).contiguous()

        if reverse == False:
            y = [y[:, i:i+1] for i in range(y.shape[1])]

            nll = 0.0
            objective = 0.0
            z = []
            
            for i in range(len(y)):
                yi = y[i]

                yi = yi.view(B, s2y, s1y, s3y)
                yi = yi.permute(0, 2, 1, 3).contiguous()

                if self.residual:
                    y_dif = yi - x.permute(0, 3, 2, 1)[:, :, -1:]
                    zi, logdet = self.flow(x, y_dif, logdet=logdet, reverse=False)
                else:
                    zi, logdet = self.flow(x, yi, logdet=logdet, reverse=False)
                
                z.append(zi)

                x = torch.cat((x[:, :, 1:], yi.permute(0, 3, 2, 1)), dim=2)

                objective += GaussianDiag.logp(mean, logs, zi)
                
            mu, logsigma = self.nn_theta(z[0], z[1])
            objective += GaussianDiag.logp(mu, logsigma, z[1])

            mu, logsigma = self.nn_theta(z[1], z[2])
            objective += GaussianDiag.logp(mu, logsigma, z[2])
            
            objective += logdet    
            nll = -objective / float(np.log(2.) * dimensions)

            z_out = zi.clone()
                
            return z_out, nll

        else:
            with torch.no_grad():
                if reverse:
                    ys = []
                    #temperature = 0.6 #0.5
                    eps_std = 0.006
                    z = []
                    z_init = GaussianDiag.batchsample(B, mean, logs, eps_std)
                    
                    y0 = x[:, :, -2:-1].permute(0, 3, 2, 1)
                    y1 = x[:, :, -1:].permute(0, 3, 2, 1)
                    x0 = torch.cat((x[:, :, :2], x[:, :, :-2]), dim=2)
                    x1 = torch.cat((x[:, :, :1], x[:, :, :-1]), dim=2)
                    if self.residual:
                        y_dif = y0 - x0.permute(0, 3, 2, 1)[:, :, -1:]
                        y_dif = y1 - x1.permute(0, 3, 2, 1)[:, :, -1:]
                        z1, logdet_i = self.flow(x1, y_dif, logdet=logdet, reverse=False)
                    else:
                        z0, logdet_i = self.flow(x0, y0, logdet=logdet, reverse=False)
                        z1, logdet_i = self.flow(x1, y1, logdet=logdet, reverse=False)
                    
                    z.append(z1)
                    
                    for i in range(self.pred_length):    
                        mu, logsigma = self.nn_theta(z[i], z_init)
                        zi = GaussianDiag.sample(mu, logsigma, eps_std)
                        logdet += logdet_i
                        if self.residual:
                            y_dif, logdet_i = self.flow(x, zi, eps_std=eps_std, reverse=True)
                        else:                            
                            yi, logdet_i= self.flow(x, zi, eps_std=eps_std, reverse=True)

                        z.append(zi)

                        if self.residual:
                            yi = x.permute(0, 3, 2, 1)[:, :, -1:] + y_dif
                        
                        x = torch.cat((x[:, :, 1:], yi.permute(0, 3, 2, 1)), dim=2)
                        
                        ys.append(yi)
   
                    y = torch.cat(ys, dim=2)
                    y = y.permute(0, 2, 1, 3).contiguous()
                    y = y.view(B, self.pred_length, -1)

            return y, logdet

    def prior(self):
        if self.learn_top:
            return self.new_mean, self.new_logs
        else:
            return torch.zeros_like(self.new_mean), torch.zeros_like(self.new_mean)


################################
class MotionFlow(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layers = nn.ModuleList()
        self.output_shapes = []
        x_size = args.x_size
        y_size = args.y_size
        x_hidden_channels = args.x_hidden_channels
        x_hidden_size = args.x_hidden_size
        y_hidden_channels = args.y_hidden_channels
        K = args.flow_depth

        all_masks, params = mask_param(x_size, args)

        C, H, W = y_size

        # K CGlowStep
        for k in range(0, K):
            self.layers.append(structuredGlowStep(x_size=x_size, y_size=y_size, x_hidden_channels=x_hidden_channels, x_hidden_size=x_hidden_size, y_hidden_channels=y_hidden_channels, masks=all_masks, params=params))
            self.output_shapes.append([-1, C, H, W])
     
        self.x_convs = x_convs(x_size=x_size, y_size=y_size, x_hidden_channels=x_hidden_channels, x_hidden_size=x_hidden_size, y_hidden_channels=y_hidden_channels, masks=all_masks, params=params)


    def forward(self, x, y, logdet=0.0, reverse=False, eps_std=1.0):
        if reverse == False:
            return self.encode(x, y, logdet)
        else:
            return self.decode(x, y, logdet, eps_std)

    def encode(self, x, y, logdet=0.0):
        x_actnorm, x_invconv = self.x_convs(x)
        for layer, shape in zip(self.layers, self.output_shapes):
            y, logdet = layer(x, y, logdet, reverse=False, x_actnorm=x_actnorm, x_invconv=x_invconv)
        return y, logdet

    def decode(self, x, y, logdet=0.0, eps_std=1.0):
        x_actnorm, x_invconv = self.x_convs(x)
        for layer in reversed(self.layers):
            y, logdet = layer(x, y, logdet=logdet, reverse=True, x_actnorm=x_actnorm, x_invconv=x_invconv)
        return y, logdet


################################
class x_convs(nn.Module):
    def __init__(self, x_size, y_size, x_hidden_channels, x_hidden_size, y_hidden_channels, masks, params):
        super().__init__()

        C_x,H_x,W_x = x_size
        
        self.mask_init1, self.mask_undilated1, self.mask_dilated1 = masks[0]
        self.mask_init2, self.mask_undilated2, self.mask_dilated2 = masks[1]

        max_dilation = params['max_dilation']
        input_channels = params['input_channels']
        conv_bias = params['conv_bias']
        conv_mask_weight = params['conv_mask_weight']
        nr_filters = params['nr_filters']
        kernel_size = [params['kernel_size'], params['kernel_size']]
        dropout = nn.Dropout2d(0.5)

        # conditioning networks
        self.x_Con_1 = nn.Sequential(
            locally_masked_conv2d(input_channels + 1, nr_filters, self.mask_init1, kernel_size=kernel_size, bias=conv_bias, mask_weight=conv_mask_weight), PONO(), concat_elu(), dropout,
            locally_masked_conv2d(2*nr_filters, nr_filters//2, self.mask_undilated1, kernel_size=kernel_size, bias=conv_bias, mask_weight=conv_mask_weight), PONO(), concat_elu(),
            locally_masked_conv2d(nr_filters, 4, self.mask_dilated1, kernel_size=kernel_size, dilation=max_dilation, bias=conv_bias, mask_weight=conv_mask_weight), PONO(), concat_elu())

        self.x_Con_2 = nn.Sequential(
            locally_masked_conv2d(input_channels + 1, nr_filters, self.mask_init2, kernel_size=kernel_size, bias=conv_bias, mask_weight=conv_mask_weight), PONO(), concat_elu(), dropout,
            locally_masked_conv2d(2*nr_filters, nr_filters//2, self.mask_undilated2, kernel_size=kernel_size, bias=conv_bias, mask_weight=conv_mask_weight), PONO(), concat_elu(),
            locally_masked_conv2d(nr_filters, 4, self.mask_dilated2, kernel_size=kernel_size, dilation=max_dilation, bias=conv_bias, mask_weight=conv_mask_weight), PONO(), concat_elu())

        self.x_Linear = nn.Sequential(
            LinearZeros(2*x_hidden_channels*H_x*W_x//(8*8), x_hidden_size//2), concat_elu(),
            LinearZeros(x_hidden_size, x_hidden_size//2), concat_elu())

    def forward(self, x):
        B,C,H,W = x.size()

        if x.shape[1] != 4:
            xs = [int(ss) for ss in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False).to(x.device)
            x = torch.cat((x, padding), 1)
        
        x_1 = self.x_Con_1(x)
        x_2 = self.x_Con_2(x)

        x_conv = x_1 * x_2

        a, b = torch.chunk(x_conv, 2, dim=1)
        a, _, __ = pono(a)
        c3 = a * torch.sigmoid(b)
        x = x + c3
        
        x = x.view(B, -1)
        x_actnorm = self.x_Linear(x)
        x_invconv = torch.clone(x_actnorm)

        return x_actnorm, x_invconv


################################
class structuredGlowStep(nn.Module):
    def __init__(self, x_size, y_size, x_hidden_channels, x_hidden_size, y_hidden_channels, masks, params):
        super().__init__()

        # 1. cond-actnorm
        self.actnorm = CondActNorm(x_size=x_size, y_channels=y_size[0], x_hidden_channels=x_hidden_channels, x_hidden_size=x_hidden_size, masks=masks, params=params)
        # 2. cond-1x1conv
        self.invconv = Cond1x1Conv(x_size=x_size, x_hidden_channels=x_hidden_channels, x_hidden_size=x_hidden_size, y_channels=y_size[0], masks=masks, params=params)
        # 3. cond-affine
        self.affine = CondAffineCoupling(x_size=x_size, y_size=[y_size[0] // 2, y_size[1], y_size[2]], hidden_channels=y_hidden_channels, masks=masks, params=params)

    def forward(self, x, y, logdet=None, reverse=False, x_actnorm=None, x_invconv=None):
        if reverse is False:
            # 1. cond-actnorm
            y, logdet = self.actnorm(x_actnorm, y, logdet, reverse=False)
            # 2. cond-1x1conv
            y, logdet = self.invconv(x_invconv, y, logdet, reverse=False)
            # 3. cond-affine
            y, logdet = self.affine(x, y, logdet, reverse=False)
            # Return

            return y, logdet
        else:
            # 3. cond-affine
            y, logdet = self.affine(x, y, logdet, reverse=True)
            # 2. cond-1x1conv
            y, logdet = self.invconv(x_invconv, y, logdet, reverse=True)
            # 1. cond-actnorm
            y, logdet = self.actnorm(x_actnorm, y, logdet, reverse=True)
            # Return

            return y, logdet


################################
class CondActNorm(nn.Module):
    def __init__(self, x_size, y_channels, x_hidden_channels, x_hidden_size, masks, params):
        super().__init__()

        self.x_Linear = nn.Sequential(LinearZeros(x_hidden_size, 2*y_channels), nn.Tanh())

    def forward(self, x, y, logdet=0, reverse=False):
        B = x.size(0)

        x = self.x_Linear(x)
        x = x.view(B, -1, 1, 1)

        logs, bias = split_feature(x)
        dimentions = y.size(2) * y.size(3)

        if not reverse:
            # center and scale
            y = y + bias
            y = y * torch.exp(logs)
            dlogdet = dimentions * torch.sum(logs, dim=(1,2,3))
            logdet = logdet + dlogdet
        else:
            # scale and center
            y = y * torch.exp(-logs)
            y = y - bias
            dlogdet = - dimentions * torch.sum(logs, dim=(1,2,3))
            logdet = logdet + dlogdet

        return y, logdet
        
        
################################
class Cond1x1Conv(nn.Module):
    def __init__(self, x_size, x_hidden_channels, x_hidden_size, y_channels, masks, params):
        super().__init__()
        self.x_Linear = nn.Sequential(LinearNorm(x_hidden_size, y_channels*y_channels), nn.Tanh())

    def get_weight(self, x, y, reverse):
        y_channels = y.size(1)
        B = x.size(0)

        x = self.x_Linear(x)
        x = x.view(B, -1, 1, 1)
        
        weight = x.view(B, y_channels, y_channels)

        dimensions = y.size(2) * y.size(3)
        dlogdet = torch.slogdet(weight)[1] * dimensions

        if reverse == False:
            weight = weight.view(B, y_channels, y_channels,1,1)
        else:
            weight = torch.inverse(weight.double()).float().view(B, y_channels, y_channels,1,1)

        return weight, dlogdet

    def forward(self, x, y, logdet=None, reverse=False):
        weight, dlogdet = self.get_weight(x, y, reverse)
        B,C,H,W = y.size()
        y = y.contiguous().view(1, B*C, H, W)
        B_k, C_i_k, C_o_k, H_k, W_k = weight.size()
        assert B == B_k and C == C_i_k and C == C_o_k, "The input and kernel dimensions are different"
        weight = weight.contiguous().view(B_k * C_i_k, C_o_k, H_k, W_k)#.double()

        if reverse == False:
            z = F.conv2d(y, weight, groups=B)
            z = z.view(B,C,H,W)
            if logdet is not None:
                logdet = logdet + dlogdet

            return z, logdet
        else:
            z = F.conv2d(y, weight, groups=B)
            z = z.view(B,C,H,W)

            if logdet is not None:
                logdet = logdet - dlogdet

            return z, logdet


################################
class CondAffineCoupling(nn.Module):
    def __init__(self, x_size, y_size, hidden_channels, masks, params):
        super().__init__()
        
        self.mask_init1, self.mask_undilated1, self.mask_dilated1 = masks[0]
        self.mask_init2, self.mask_undilated2, self.mask_dilated2 = masks[1]

        input_channels = params['input_channels']
        conv_bias = params['conv_bias']
        conv_mask_weight = params['conv_mask_weight']
        nr_filters = params['nr_filters']
        kernel_size = [params['kernel_size'], params['kernel_size']]
        dropout = nn.Dropout2d(0.5)

        self.resize_x_1 = nn.Sequential(
            locally_masked_conv2d(input_channels + 1, nr_filters, self.mask_init1, kernel_size=kernel_size, bias=conv_bias, mask_weight=conv_mask_weight), PONO(), concat_elu(), dropout,
            locally_masked_conv2d(2*nr_filters, nr_filters//2, self.mask_undilated1, kernel_size=kernel_size, bias=conv_bias, mask_weight=conv_mask_weight), PONO(), concat_elu(),
            locally_masked_conv2d(nr_filters, 4, self.mask_dilated1, kernel_size=kernel_size, bias=conv_bias, mask_weight=conv_mask_weight), PONO(), concat_elu())

        self.resize_x_2 = nn.Sequential(
            locally_masked_conv2d(input_channels + 1, nr_filters, self.mask_init2, kernel_size=kernel_size, bias=conv_bias, mask_weight=conv_mask_weight), PONO(), concat_elu(), dropout,
            locally_masked_conv2d(2*nr_filters, nr_filters//2, self.mask_undilated2, kernel_size=kernel_size, bias=conv_bias, mask_weight=conv_mask_weight), PONO(), concat_elu(), 
            locally_masked_conv2d(nr_filters, 4, self.mask_dilated2, kernel_size=kernel_size, bias=conv_bias, mask_weight=conv_mask_weight), PONO(), concat_elu())
    
        self.resize_x = nn.Sequential(
            Conv2dZeros(4, 16//2), concat_elu(), 
            Conv2dResize((16,x_size[1],x_size[2]), out_size=y_size), concat_elu(), 
            Conv2dZeros(y_size[0]*2, y_size[0]//2), concat_elu())
    
        self.f = nn.Sequential(
            Conv2dNormy(y_size[0]*2, hidden_channels//2), concat_elu(), 
            Conv2dNormy(hidden_channels, hidden_channels//2, kernel_size=[1, 1]), concat_elu(), 
            Conv2dZerosy(hidden_channels, 2*y_size[0]), nn.Tanh())

    def forward(self, x, y, logdet=0.0, reverse=False):
        z1, z2 = split_feature(y, "split")
        
        if x.shape[1] != 4:
            xs = [int(ss) for ss in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.cuda() if x.is_cuda else padding
            x = torch.cat((x, self.init_padding), 1)
        
        x_1 = self.resize_x_1(x) 
        x_2 = self.resize_x_2(x)
        
        x_conv = x_1 * x_2

        a, b = torch.chunk(x_conv, 2, dim=1)
        a, _, __ = pono(a)
        c3 = a * torch.sigmoid(b)
        x = x + c3
        
        x = self.resize_x(x.float())

        h = torch.cat((x,z1), dim=1)
        h = self.f(h)

        shift, scale = split_feature(h, "cross")
        scale = torch.sigmoid(scale + 2.)
        if reverse == False:
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = torch.sum(torch.log(scale), dim=(1, 2, 3)) + logdet

        if reverse == True:
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -torch.sum(torch.log(scale), dim=(1, 2, 3)) + logdet

        z = torch.cat((z1, z2), dim=1)

        return z, logdet


################################
class Conv2dZeros(nn.Conv2d):
    def __init__(self, in_channel, out_channel, kernel_size=[3,3], stride=[1,1]):
        padding = (kernel_size[0] - 1) // 2
        super().__init__(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.weight.data.normal_(mean=0.0, std=0.1)


################################
class GaussianDiag:
    Log2PI = float(np.log(2 * np.pi))

    @staticmethod
    def likelihood(mean, logs, x):
        return -0.5 * (logs * 2. + ((x - mean) ** 2.) / torch.exp(logs * 2.) + GaussianDiag.Log2PI)

    @staticmethod
    def logp(mean, logs, x):
        likelihood = GaussianDiag.likelihood(mean, logs, x)
        return torch.sum(likelihood, dim=(1, 2, 3))

    @staticmethod
    def sample(mean, logs, eps_std=None):
        eps_std = eps_std or 1
        eps = torch.normal(mean=torch.zeros_like(mean), std=torch.ones_like(logs) * eps_std)
        return mean + torch.exp(logs) * eps

    @staticmethod
    def batchsample(batchsize, mean, logs, eps_std=None):
        eps_std = eps_std or 1
        sample = GaussianDiag.sample(mean, logs, eps_std)
        for i in range(1, batchsize):
            s = GaussianDiag.sample(mean, logs, eps_std)
            sample = torch.cat((sample, s), dim=0)
        return sample


################################
class Conv2dResize(nn.Conv2d):
    def __init__(self, in_size, out_size):

        stride = [in_size[1]//out_size[1], in_size[2]//out_size[2]]
        kernel_size = Conv2dResize.compute_kernel_size(in_size, out_size, stride)
        super().__init__(in_channels=in_size[0], out_channels=out_size[0], kernel_size=kernel_size, stride=stride)
        self.weight.data.zero_()

    @staticmethod
    def compute_kernel_size(in_size, out_size, stride):
        k0 = in_size[1] - (out_size[1] - 1) * stride[0]
        k1 = in_size[2] - (out_size[2] - 1) * stride[1]
        return [k0, k1]


################################
class Conv1dResize(nn.Conv1d):
    def __init__(self, in_size, out_size, kernel_size, stride):

        super().__init__(in_channels=in_size[0], out_channels=out_size[0], kernel_size=kernel_size, stride=stride)
        self.weight.data.zero_()

    @staticmethod
    def compute_kernel_size(in_size, out_size, stride):
        k0 = in_size[1] - (out_size[1] - 1) * stride[0]
        k1 = in_size[2] - (out_size[2] - 1) * stride[1]
        return[k0,k1]


################################
class LinearZeros(nn.Linear):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input.float())
        return output

        
################################
class LinearNorm(nn.Linear):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.weight.data.normal_(mean=0.0, std=0.1)
        self.bias.data.normal_(mean=0.0, std=0.1)


################################
class Conv2dZerosy(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=[3, 3], stride=[1, 1]):
        padding = [(kernel_size[0]-1)//2, (kernel_size[1]-1)//2]
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)

        self.logscale_factor = 3.0
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        self.register_parameter("newbias", nn.Parameter(torch.zeros(out_channels, 1, 1)))

        # init
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        output = output + self.newbias
        output = output * torch.exp(self.logs * self.logscale_factor)
        return output


################################
class Conv2dNormy(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=[3, 3], stride=[1, 1]):
        padding = [(kernel_size[0]-1)//2, (kernel_size[1]-1)//2]
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=False)

        # initialize weight
        self.weight.data.normal_(mean=0.0, std=0.05)
        self.actnorm = ActNorm(out_channels)

    def forward(self, input):
        x = super().forward(input)
        x,_ = self.actnorm(x)
        return x


################################
class ActNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        size = [1, num_channels, 1, 1]

        bias = torch.normal(mean=torch.zeros(*size), std=torch.ones(*size)*0.05)
        logs = torch.normal(mean=torch.zeros(*size), std=torch.ones(*size)*0.05)
        self.register_parameter("bias", nn.Parameter(torch.Tensor(bias), requires_grad=True))
        self.register_parameter("logs", nn.Parameter(torch.Tensor(logs), requires_grad=True))

    def forward(self, input, logdet=0, reverse=False):
        dimentions = input.size(1) * input.size(2) * input.size(3)
        if reverse == False:
            input = input + self.bias
            input = input * torch.exp(self.logs)
            dlogdet = torch.sum(self.logs) * dimentions
            logdet = logdet + dlogdet

        if reverse == True:
            input = input * torch.exp(-self.logs)
            input = input - self.bias
            dlogdet = - torch.sum(self.logs) * dimentions
            logdet = logdet + dlogdet

        return input, logdet


################################
class NNTheta(nn.Module):
    def __init__(self, encoder_ch_in, encoder_mode, num_blocks, h_ch_in=None):
        super(NNTheta, self).__init__()
        self.encoder_mode = encoder_mode

        if h_ch_in is not None:
            self.conv1 = nn.Conv2d(in_channels=h_ch_in, out_channels=h_ch_in, kernel_size=1)
            initialize(self.conv1, mode='gaussian')

        dilations = [1, 2, 4]
        self.latent_encoder = nn.ModuleList()
        for i in range(num_blocks):
            self.latent_encoder.append(nn.ModuleList(
                [self.latent_dist_encoder(encoder_ch_in, dilation=d, mode=encoder_mode) for d in dilations]))

        if h_ch_in:
            self.conv2 = nn.Conv2d(in_channels=encoder_ch_in, out_channels=encoder_ch_in, kernel_size=1)
            initialize(self.conv2, mode='zeros')
        else:
            self.conv2 = nn.Conv2d(in_channels=encoder_ch_in, out_channels=2 * encoder_ch_in, kernel_size=1)
            initialize(self.conv2, mode='zeros')

    def forward(self, z_past, h):
        if h is not None:
            h = self.conv1(h)
            encoder_input = torch.cat([z_past, h], dim=1) 
        else:
            encoder_input = z_past.clone()

        for block in self.latent_encoder:
            parallel_outs = [pb(encoder_input) for pb in block]

            parallel_outs.append(encoder_input)
            encoder_input = sum(parallel_outs)

        last_t = self.conv2(encoder_input)
        deltaz_t, logsigma_t = last_t[:, 0::2, ...], last_t[:, 1::2, ...]

        logsigma_t = torch.clamp(logsigma_t, min=-15., max=15.)
        mu_t = deltaz_t + z_past
        return mu_t, logsigma_t

    @staticmethod
    def latent_dist_encoder(ch_in, dilation, mode):

        if mode == "conv_net":
            layer1 = nn.Conv2d(in_channels=ch_in, out_channels=512, kernel_size=(3, 3),
                               dilation=(dilation, dilation), padding=(dilation, dilation))
            initialize(layer1, mode='gaussian')
            layer2 = GATU2D(channels=512)
            layer3 = nn.Conv2d(in_channels=512, out_channels=ch_in, kernel_size=(3, 3),
                               dilation=(dilation, dilation), padding=(dilation, dilation))
            initialize(layer3, mode='zeros')

            block = nn.Sequential(*[layer1, nn.ReLU(inplace=True), layer2, layer3])

        return block


################################
class GATU2D(nn.Module):
    def __init__(self, channels):
        super(GATU2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        initialize(self.conv1, mode='gaussian')
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        initialize(self.conv2, mode='gaussian')

    def forward(self, x):
        out1 = torch.tanh(self.conv1(x))
        out2 = torch.sigmoid(self.conv2(x))
        return out1 * out2


################################
def initialize(layer, mode):
    if mode == 'gaussian':
        nn.init.normal_(layer.weight, 0., 0.05)
        nn.init.normal_(layer.bias, 0., 0.05)

    elif mode == 'zeros':
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)

    else:
        raise NotImplementedError("To be implemented")


################################
def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...] 
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]  # [start:stop:step] 