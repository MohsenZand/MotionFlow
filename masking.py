import os
import math
import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm as wn
import torch.nn.functional as F


#######################################
class locally_masked_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, mask, kernel_size=(3,3), dilation=1, bias=True, mask_weight=False):
        """A memory-efficient implementation of Locally Masked Convolution.
        https://github.com/ajayjain/lmconv

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (tuple): Size of the convolving kernel as a tuple of two ints.
                Default: (3, 3). The first int is used for the height dimension, and the second int for the width dimension.
            dilation (int): Spacing between kernel elements. Default: 1
            bias (bool): If True, adds a learnable bias to the output. Default: True
            mask_weight (bool): If True, adds a learnable weight to condition the layer on the mask. Default: False
        """
        super(locally_masked_conv2d, self).__init__()

        self.mask = mask 

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation

        # Pad to maintain spatial dimensions
        pad0 = (dilation * (kernel_size[0] - 1)) // 2
        pad1 = (dilation * (kernel_size[1] - 1)) // 2
        self.padding = (pad0, pad1)

        # Conv parameters
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.mask_weight = nn.Parameter(torch.Tensor(out_channels, *kernel_size)) if mask_weight else None
        self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        # Adapted from PyTorch _ConvNd implementation
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.mask_weight is not None:
            nn.init.kaiming_uniform_(self.mask_weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, mask=None):
        if not x.is_cuda:
            self.mask = self.mask.cpu()
        return _locally_masked_conv2d.apply(x, self.mask, self.weight, self.mask_weight, self.bias, self.dilation, self.padding)



#######################################
class _locally_masked_conv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, weight, mask_weight=None, bias=None, dilation=1, padding=1):
        assert len(x.shape) == 4, "Unfold/fold only support 4D batched image-like tensors"
        ctx.save_for_backward(x, mask, weight, mask_weight)
        ctx.dilation = dilation
        ctx.padding = padding
        ctx.H, ctx.W = x.size(2), x.size(3)

        # Shapes
        ctx.output_shape = (x.shape[2], x.shape[3])
        out_channels, in_channels, k1, k2 = weight.shape
        assert x.size(1) == in_channels
        assert mask.size(1) == k1 * k2

        # Step 1: Unfold (im2col)
        x = F.unfold(x, (k1, k2), dilation=dilation, padding=padding)

        # Step 2: Mask x. Avoid repeating mask in_channels times by reshaping x_unf (memory efficient)
        assert x.size(1) % in_channels == 0
        x_unf_channels_batched = x.view(x.size(0) * in_channels, x.size(1) // in_channels, x.size(2))
        x = torch.mul(x_unf_channels_batched, mask).view(x.shape)

        # Step 3: Perform convolution via matrix multiplication and addition
        weight_matrix = weight.view(out_channels, -1)

        x = weight_matrix.matmul(x.float())
        
        if bias is not None:
            x = x + bias.unsqueeze(0).unsqueeze(2)

        # Step 4: Apply weight on mask, if provided. Equivalent to concatenating x and mask.
        if mask_weight is not None:
            x = x + mask_weight.view(out_channels, -1).matmul(mask)

        # Step 4: Restore shape
        output = x.view(x.size(0), x.size(1), *ctx.output_shape)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, mask, weight, mask_weight = ctx.saved_tensors
        out_channels, in_channels, k1, k2 = weight.shape
        grad_output_unfolded = grad_output.view(grad_output.size(0), grad_output.size(1), -1)  # B x C_out x (H*W)

        # Compute gradients
        grad_x = grad_weight = grad_mask_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            weight_ = weight.view(out_channels, -1)
            grad_x_ = weight_.transpose(0, 1).matmul(grad_output_unfolded)
            grad_x_shape = grad_x_.shape
            # View to allow masking, since mask needs to be broadcast C_in times
            assert grad_x_.size(1) % in_channels == 0
            grad_x_ = grad_x_.view(grad_x_.size(0) * in_channels, grad_x_.size(1) // in_channels, grad_x_.size(2))
            grad_x_ = torch.mul(grad_x_, mask).view(grad_x_shape)
            grad_x = F.fold(grad_x_, (ctx.H, ctx.W), (k1, k2), dilation=ctx.dilation, padding=ctx.padding)
        if ctx.needs_input_grad[2]:
            # Recompute unfold and masking to avoid storing unfolded x, at the cost of extra compute
            x_ = F.unfold(x, (k1, k2), dilation=ctx.dilation, padding=ctx.padding)  # B x 27 x 64
            x_unf_shape = x_.shape
            assert x_.size(1) % in_channels == 0
            x_ = x_.view(x_.size(0) * in_channels, x_.size(1) // in_channels, x_.size(2))
            x_ = torch.mul(x_, mask).view(x_unf_shape)

            grad_weight = grad_output_unfolded.matmul(x_.transpose(2, 1))
            grad_weight = grad_weight.view(grad_weight.size(0), *weight.shape)
        if ctx.needs_input_grad[3]:
            grad_mask_weight = grad_output_unfolded.matmul(mask.transpose(2, 1))  # B x C_out x k1*k2
            grad_mask_weight = grad_mask_weight.view(grad_mask_weight.size(0), *mask_weight.shape)
        if ctx.needs_input_grad[4]:
            grad_bias = grad_output.sum(dim=(0, 2, 3))

        assert not ctx.needs_input_grad[1], "Can't differentiate wrt mask"

        return grad_x, None, grad_weight, grad_mask_weight, grad_bias, None, None


#######################################
class PONO(nn.Module):
    def forward(self, x):
        x, _, __ = pono(x)
        return x


#######################################
class concat_elu(nn.Module):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    def forward(self, x, mask=None):
        # Pytorch ordering
        axis = len(x.size()) - 3
        return F.elu(torch.cat([x, -x], dim=axis), inplace=True)


#######################################
def mask_param(seq_size, args):
    plot_mask = args.plot_mask
    param = {}
    param['max_dilation'] = args.max_dilation
    observed_idx = args.observed_idx
    orders = [args.order1, args.order2]
    param['nr_logistic_mix'] = args.nr_logistic_mix
    param['input_channels'] = args.input_channels
    param['conv_bias'] = args.conv_bias
    param['conv_mask_weight'] = args.conv_mask_weight
    param['nr_filters'] = args.nr_filters
    param['feature_norm_op'] = args.feature_norm_op
    
    param['num_mix'] = args.num_mix
    param['kernel_size'] = args.kernel_size

    if args.device != 'cpu':
        device = int(args.device)

    all_generation_idx =[]
    for order in range(2):
        generation_idx = get_generation_order_idx(orders[order], seq_size[1], seq_size[2])
        all_generation_idx.append(generation_idx)

    all_masks = []
    for _, generation_idx in enumerate(all_generation_idx):
        masks = get_masks(generation_idx, seq_size[1], seq_size[2], param['kernel_size'], 
                        param['max_dilation'], observed_idx=observed_idx, out_dir=args.run_dir, 
                        plot_suffix=f"obs{size2str(seq_size)}", plot=plot_mask)
        if args.device != 'cpu':
            masks = [mask for mask in masks]
        all_masks.append(masks)

    return all_masks, param


#######################################
def pono(x, epsilon=1e-5):
    """Positional normalization"""
    mean = x.mean(dim=1, keepdim=True)
    std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
    output = (x - mean) / std
    return output, mean, std


#######################################
def get_generation_order_idx(order, rows, cols):
    """Get (rows*cols) x 2 np array given order that pixels are generated"""
    assert order in ["LR", "TB", "s_curve"]
    return eval(f"{order}_idx")(rows, cols)


#######################################
def TB_idx(rows, cols):
    """Generate top-bottom ordering """
    idx = []
    for f in range(rows):
        fr_idx = range(cols) 
        for j in fr_idx:
            idx.append((f, j))
    return np.array(idx)


#######################################
def LR_idx(rows, cols):
    """Generate left-right ordering"""
    idx = []
    for j in range(cols):
        jnt_idx = range(rows) 
        for f in jnt_idx:
            idx.append((f, j))
    return np.array(idx)


#######################################
def s_curve_idx(rows, cols):
    """Generate S shape curve"""
    idx = []
    for r in range(rows):
        col_idx = range(cols) if r % 2 == 0 else range(cols-1, -1, -1)
        for c in col_idx:
            idx.append((r, c))
    return np.array(idx)


#######################################
def size2str(obs):
    return 'x'.join(map(str, obs))


#######################################
def get_masks(generation_idx, nrows, ncols, k=3, max_dilation=1, observed_idx=None, out_dir="/tmp", plot_suffix="", plot=True):
    """Get and plot three masks: mask type A for first layer, mask type B for later layers, and mask type B with dilation.
    Masks are copied to GPU and repeated along the batch dimension torch.cuda.device_count() times for DataParallel support."""
    mask_init = get_unfolded_masks(generation_idx, nrows, ncols, k=k, dilation=1, mask_type='A', observed_idx=observed_idx)
    mask_undilated = get_unfolded_masks(generation_idx, nrows, ncols, k=k, dilation=1, mask_type='B', observed_idx=observed_idx)
    if plot:
        plot_unfolded_masks(nrows, ncols, generation_idx, mask_init, k=k, out_path=os.path.join(out_dir, f"mask_init_{plot_suffix}.pdf"))
        plot_unfolded_masks(nrows, ncols, generation_idx, mask_undilated, k=k, out_path=os.path.join(out_dir, f"mask_undilated_{plot_suffix}.pdf"))
    mask_init = mask_init.cuda(non_blocking=True)  # .repeat(torch.cuda.device_count(), 1, 1)
    mask_undilated = mask_undilated.cuda(non_blocking=True)  # .repeat(torch.cuda.device_count(), 1, 1)

    if max_dilation == 1:
        mask_dilated = mask_undilated
    else:
        mask_dilated = get_unfolded_masks(generation_idx, nrows, ncols, k=k, dilation=max_dilation, mask_type='B', observed_idx=observed_idx)
        if plot:
            plot_unfolded_masks(nrows, ncols, generation_idx, mask_dilated, k=k, out_path=os.path.join(out_dir, f"mask_dilated_d{max_dilation}_{plot_suffix}.pdf"))
        mask_dilated = mask_dilated.cuda(non_blocking=True)  # .repeat(torch.cuda.device_count(), 1, 1)

    return mask_init, mask_undilated, mask_dilated


#######################################
def get_unfolded_masks(generation_order_idx, nrows, ncols, k=3, dilation=1, mask_type='B', observed_idx=None):
    assert mask_type in ['A', 'B']
    masks = kernel_masks(generation_order_idx, nrows, ncols, k, dilation, mask_type, set_padding=0, observed_idx=observed_idx)
    masks = torch.tensor(masks, dtype=torch.float)
    masks_unf = masks.view(1, nrows * ncols, -1).transpose(1, 2)
    return masks_unf


#######################################
def kernel_masks(generation_order_idx, nrows, ncols, k=3, dilation=1, mask_type='B', set_padding=0, observed_idx=None) -> np.ndarray:
    """Generate kernel masks given a pixel generation order.
    
    Args:
        generation_order_idx: N x 2 array, order to generate pixels. 
        nrows
        ncols
        k
        dilation
        mask_type: A or B
        set_padding
        observed_idx: M x 2 array, for coords in this list, will allow all locations to condition.
            Useful for inpainting tasks, where some context is observed and masking is only needed in the unobserved region.
    """
    assert k % 2 == 1, "Only odd sized kernels are implemented"
    half_k = int(k / 2)
    masks = np.zeros((len(generation_order_idx), k, k))

    locs_generated = set()
    if observed_idx is not None:
        # Can observe some context
        for r, c in observed_idx:
            locs_generated.add((r, c))

    # Set masks
    for i, (r, c) in enumerate(generation_order_idx):
        row_major_index = r * ncols + c
        for dr in range(-half_k, half_k+1):
            for dc in range(-half_k, half_k+1):
                if dr == 0 and dc == 0:
                    # skip center pixel of mask
                    continue

                loc = (r + dr * dilation, c + dc * dilation)
                if loc in locs_generated:
                    # The desired location has been generated,
                    # so we can condition on it
                    masks[row_major_index, half_k + dr, half_k + dc] = 1
                elif not (0 <= loc[0] < nrows and 0 <= loc[1] < ncols):
                    # Kernel location overlaps with padding
                    masks[row_major_index, half_k + dr, half_k + dc] = set_padding
        locs_generated.add((r, c))

    if mask_type == 'B':
        masks[:, half_k, half_k] = 1
    else:
        assert np.all(masks[:, half_k, half_k] == 0)

    return masks


#######################################
def plot_unfolded_masks(nrows, ncols, generation_order, unfolded_masks, k=3, out_path=None):
    masks = unfolded_masks.view(k, k, -1).permute(2, 0, 1)
    print(f"Plotting kernel masks and saving to {out_path}...")
    plot_masks(nrows, ncols, generation_order, masks, k=3, out_path=out_path)


#######################################
def plot_masks(nrows, ncols, generation_order, masks, k=3, out_path=None):
    fig, axes = plt.subplots(nrows, ncols)
    plt.suptitle(f"Kernel masks")
    for row_major_index, ((r, c), mask) in enumerate(zip(generation_order, masks)):
        axes[row_major_index // ncols, row_major_index % ncols].imshow(mask, vmin=0, vmax=1)
    plt.setp(axes, xticks=[], yticks=[])
    if out_path:
        plt.savefig(out_path)
    else:
        plt.show()


#######################################
def plot_order(generation_idx, obs, out_path=None):
    """Plot generation coordinate list. A star on the curve
    denotes the pixel generated last. obs is a three-tuple of input image dimensions,
    (input-channels-unused, num_rows, num_cols)"""

    plt.figure(figsize=(10, 5))
    plt.hlines(np.arange(-1, obs[2])+0.5, xmin=-0.5, xmax=obs[1]-0.5, alpha=0.5)
    plt.vlines(np.arange(-1, obs[1])+0.5, ymin=-0.5, ymax=obs[2]-0.5, alpha=0.5)
    cols, rows = zip(*generation_idx)
    plt.plot(cols, rows, color="b")
    plt.scatter([cols[0]], [rows[0]], marker="*", s=500, c="r")
    plt.scatter([cols[-1]], [rows[-1]], marker="o", s=500, c="r")
    plt.xticks(np.arange(obs[1]))
    plt.axis("equal")
    plt.xlabel('rows')
    plt.ylabel('cols')
    plt.gca().invert_yaxis()
    if out_path:
        plt.savefig(out_path + 'mask.jpg')
    else:
        plt.show()

    