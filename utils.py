import os
import datetime
import glob
import re
from pathlib import Path
import pickle 
import numpy as np 
import chumpy as ch
from chumpy.ch import MatVecMult
import torch
import cv2 



def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'MotionFlow 🚀 {date_modified()} torch {torch.__version__} '  # string
    device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'
    print(s)

    return torch.device('cuda:0' if cuda else 'cpu')


def date_modified(path=__file__):
    # return human-readable file modification date, i.e. '2021-3-26'
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    # https://github.com/ultralytics/yolov5
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def save_model(model, optim, scheduler, dir, iteration, epoch, iter_save=False):
    if iter_save:
        path = os.path.join(dir, "checkpoint_{}.pth.tar".format(iteration))
    else:
        path = os.path.join(dir, "last.pth.tar")
    state = {}
    state["iteration"] = iteration
    state['epoch'] = epoch
    state["modelname"] = model.__class__.__name__
    state["model"] = model.state_dict()
    state["optim"] = optim.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    else:
        state["scheduler"] = None

    torch.save(state, path)


def load_state(path, cuda):
    if cuda:
        print ("load to gpu")
        state = torch.load(path)
    else:
        print ("load to cpu")
        state = torch.load(path, map_location=lambda storage, loc: storage)

    return state


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask].astype(int), minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

    
def compute_accuracy(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc 


def get_closest_rotmat(rotmats):
    """
    Finds the rotation matrix that is closest to the inputs in terms of the Frobenius norm. For each input matrix
    it computes the SVD as R = USV' and sets R_closest = UV'. Additionally, it is made sure that det(R_closest) == 1.
    Args:
        rotmats: np array of shape (..., 3, 3).

    Returns:
        A numpy array of the same shape as the inputs.
    """
    u, s, vh = np.linalg.svd(rotmats)
    r_closest = np.matmul(u, vh)

    # if the determinant of UV' is -1, we must flip the sign of the last column of u
    det = np.linalg.det(r_closest)  # (..., )
    iden = eye(3, det.shape)
    iden[..., 2, 2] = np.sign(det)
    r_closest = np.matmul(np.matmul(u, iden), vh)
    return r_closest


def eye(n, batch_shape):
    iden = np.zeros(np.concatenate([batch_shape, [n, n]]))
    iden[..., 0, 0] = 1.0
    iden[..., 1, 1] = 1.0
    iden[..., 2, 2] = 1.0
    return iden   


def is_valid_rotmat(rotmats, thresh=1e-6):
    """
    Checks that the rotation matrices are valid, i.e. R*R' == I and det(R) == 1
    Args:
        rotmats: A np array of shape (..., 3, 3).
        thresh: Numerical threshold.

    Returns:
        True if all rotation matrices are valid, False if at least one is not valid.
    """
    # check we have a valid rotation matrix
    rotmats_t = np.transpose(rotmats, tuple(range(len(rotmats.shape[:-2]))) + (-1, -2))
    is_orthogonal = np.all(np.abs(np.matmul(rotmats, rotmats_t) - eye(3, rotmats.shape[:-2])) < thresh)
    det_is_one = np.all(np.abs(np.linalg.det(rotmats) - 1.0) < thresh)
    return is_orthogonal and det_is_one


def sparse_to_full(joint_angles_sparse, sparse_joints_idxs, tot_nr_joints, rep="rotmat"):
    """
    Pad the given sparse joint angles with identity elements to retrieve a full skeleton with `tot_nr_joints`
    many joints.
    Args:
        joint_angles_sparse: An np array of shape (N, len(sparse_joints_idxs) * dof)
          or (N, len(sparse_joints_idxs), dof)
        sparse_joints_idxs: A list of joint indices pointing into the full skeleton given by range(0, tot_nr_joints)
        tot_nr_jonts: Total number of joints in the full skeleton.
        rep: Which representation is used, rotmat or quat

    Returns:
        The padded joint angles as an array of shape (N, tot_nr_joints*dof)
    """
    joint_idxs = sparse_joints_idxs
    assert rep in ["rotmat", "quat", "aa"]
    dof = 9 if rep == "rotmat" else 4 if rep == "quat" else 3
    n_sparse_joints = len(sparse_joints_idxs)
    angles_sparse = np.reshape(joint_angles_sparse, [-1, n_sparse_joints, dof])

    # fill in the missing indices with the identity element
    smpl_full = np.zeros(shape=[angles_sparse.shape[0], tot_nr_joints, dof])  # (N, tot_nr_joints, dof)
    if rep == "quat":
        smpl_full[..., 0] = 1.0
    elif rep == "rotmat":
        smpl_full[..., 0] = 1.0
        smpl_full[..., 4] = 1.0
        smpl_full[..., 8] = 1.0
    else:
        pass  # nothing to do for angle-axis

    smpl_full[:, joint_idxs] = angles_sparse
    smpl_full = np.reshape(smpl_full, [-1, tot_nr_joints * dof])
    return smpl_full    