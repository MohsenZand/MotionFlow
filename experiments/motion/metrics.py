import numpy as np 
import quaternion
import cv2
import copy
import tensorflow as tf 
import torch

from utils import get_closest_rotmat, is_valid_rotmat, sparse_to_full


##############################################
class MetricsEngine(object):
    '''
    https://github.com/eth-ait/spl
    Compute and aggregate various motion metrics. It keeps track of the metric values per frame, so that we can
    evaluate them for different sequence lengths.
    '''
    def __init__(self, fk_engine, target_lengths, force_valid_rot, rep, which=None):
        '''
        Initializer.
        Args:
            fk_engine: An object of type `ForwardKinematics` used to compute positions.
            target_lengths: List of target sequence lengths that should be evaluated.
            force_valid_rot: If True, the input rotation matrices might not be valid rotations and so it will find
              the closest rotation before computing the metrics.
            rep: Which representation to use, 'quat' or 'rotmat'.
            which: Which metrics to compute. Options are [mpjpe], defaults to all.
        '''
        self.which = which if which is not None else ['mpjpe']
        self.target_lengths = target_lengths
        self.force_valid_rot = force_valid_rot
        self.fk_engine = fk_engine
        self.n_samples = 0
        self._should_call_reset = False  # a guard to avoid stupid mistakes
        self.rep = rep
        assert self.rep in ['rotmat', 'quat', 'aa']

        self.metrics_agg = {k: None for k in self.which}
        self.summaries = {k: {t: None for t in target_lengths} for k in self.which}
        self.summaries['train_loss'] = None
        self.summaries['val_loss'] = None

    def reset(self):
        '''
        Reset all metrics.
        '''
        self.metrics_agg = {k: None for k in self.which}
        self.n_samples = 0
        self._should_call_reset = False  # now it's again safe to compute new values

    def compute_rotmat(self, pred_motion, targets_motion, reduce_fn='mean'):
        '''
        Compute the chosen metrics. Predictions and targets are assumed to be in rotation matrix format.
        Args:
            predictions: An np array of shape (n, seq_length, n_joints*9)
            targets: An np array of the same shape as `predictions`
            reduce_fn: Which reduce function to apply to the joint dimension, if applicable. Choices are [mean, sum].

        Returns:
            A dictionary {metric_name -> values} where the values are given per batch entry and frame as an np array
            of shape (n, seq_length). `reduce_fn` is only applied to metrics where it makes sense
            and euler angle differences.
        '''
        assert pred_motion.shape[-1] % 9 == 0, 'predictions are not rotation matrices'
        assert targets_motion.shape[-1] % 9 == 0, 'targets are not rotation matrices'
        assert reduce_fn in ['mean', 'sum']
        assert not self._should_call_reset, 'you should reset the state of this class after calling `finalize`'
        dof = 9
        n_joints = len(self.fk_engine.major_joints) 
        batch_size = pred_motion.shape[0]
        seq_length = pred_motion.shape[1]
        assert n_joints*dof == pred_motion.shape[-1], 'unexpected number of joints'

        # first reshape everything to (-1, n_joints * 9)
        pred = np.reshape(pred_motion, [-1, n_joints*dof]).copy()
        targ = np.reshape(targets_motion, [-1, n_joints*dof]).copy()

        # enforce valid rotations
        if self.force_valid_rot:
            pred_val = np.reshape(pred, [-1, n_joints, 3, 3])
            pred = get_closest_rotmat(pred_val)
            pred = np.reshape(pred, [-1, n_joints*dof])

        # check that the rotations are valid
        pred_are_valid = is_valid_rotmat(np.reshape(pred, [-1, n_joints, 3, 3]))
        assert pred_are_valid, 'predicted rotation matrices are not valid'
        targ_are_valid = is_valid_rotmat(np.reshape(targ, [-1, n_joints, 3, 3]))
        assert targ_are_valid, 'target rotation matrices are not valid'

        # add potentially missing joints
        pred = sparse_to_full(pred, self.fk_engine.major_joints, self.fk_engine.n_joints, rep='rotmat')
        targ = sparse_to_full(targ, self.fk_engine.major_joints, self.fk_engine.n_joints, rep='rotmat')

        # make sure we don't consider the root orientation
        assert pred.shape[-1] == self.fk_engine.n_joints*dof
        assert targ.shape[-1] == self.fk_engine.n_joints*dof
        pred[:, 0:9] = np.eye(3, 3).flatten()
        targ[:, 0:9] = np.eye(3, 3).flatten()

        metrics = dict()

        pred_pos = targ_pos = None

        select_joints = self.fk_engine.major_joints 
        reduce_fn_np = np.mean if reduce_fn == 'mean' else np.sum

        for metric in self.which:
            if metric == 'mpjpe':
                # compute the mean per joint position error 
                pred_local = np.reshape(pred, [-1, self.fk_engine.n_joints, 3, 3])
                targ_local = np.reshape(targ, [-1, self.fk_engine.n_joints, 3, 3])
                pred_local = rotmat2aa(pred_local)
                targ_local = rotmat2aa(targ_local)
                v = torch.mean(torch.norm(torch.tensor(targ_local.reshape((batch_size, seq_length, -1, 3))) - torch.tensor(pred_local.reshape((batch_size, seq_length, -1, 3))), dim=3), dim=2)
                #v = torch.sum(torch.mean(torch.norm(torch.tensor(targ_local.reshape((batch_size, seq_length, -1, 3))) - torch.tensor(pred_local.reshape((batch_size, seq_length, -1, 3))), dim=3), dim=2), dim=0)
                metrics[metric] = np.reshape(v.numpy(), [batch_size, seq_length])
            else:
                raise ValueError("metric '{}' unknown".format(metric))

        return metrics

    def compute_quat(self, predictions, targets, reduce_fn='mean'):
        '''
        Compute the chosen metrics. Predictions and targets are assumed to be quaternions.
        Args:
            predictions: An np array of shape (n, seq_length, n_joints*4)
            targets: An np array of the same shape as `predictions`
            reduce_fn: Which reduce function to apply to the joint dimension, if applicable. Choices are [mean, sum].

        Returns:
            A dictionary {metric_name -> values} where the values are given per batch entry and frame as an np array
            of shape (n, seq_length). `reduce_fn` is only applied to metrics where it makes sense
            and euler angle differences.
        '''
        assert predictions.shape[-1] % 4 == 0, 'predictions are not quaternions'
        assert targets.shape[-1] % 4 == 0, 'targets are not quaternions'
        assert reduce_fn in ['mean', 'sum']
        assert not self._should_call_reset, 'you should reset the state of this class after calling `finalize`'
        dof = 4
        batch_size = predictions.shape[0]
        seq_length = predictions.shape[1]

        # for simplicity we just convert quaternions to rotation matrices
        pred_q = quaternion.from_float_array(np.reshape(predictions, [batch_size, seq_length, -1, dof]))
        targ_q = quaternion.from_float_array(np.reshape(targets, [batch_size, seq_length, -1, dof]))
        pred_rots = quaternion.as_rotation_matrix(pred_q)
        targ_rots = quaternion.as_rotation_matrix(targ_q)

        preds = np.reshape(pred_rots, [batch_size, seq_length, -1])
        targs = np.reshape(targ_rots, [batch_size, seq_length, -1])
        return self.compute_rotmat(preds, targs, reduce_fn)

    def compute_aa(self, predictions, targets, reduce_fn='mean'):
        '''
        Compute the chosen metrics. Predictions and targets are assumed to be in angle-axis format.
        Args:
            predictions: An np array of shape (n, seq_length, n_joints*3)
            targets: An np array of the same shape as `predictions`
            reduce_fn: Which reduce function to apply to the joint dimension, if applicable. Choices are [mean, sum].

        Returns:
            A dictionary {metric_name -> values} where the values are given per batch entry and frame as an np array
            of shape (n, seq_length). `reduce_fn` is only applied to metrics where it makes sense
            and euler angle differences.
        '''
        assert predictions.shape[-1] % 3 == 0, 'predictions are not quaternions'
        assert targets.shape[-1] % 3 == 0, 'targets are not quaternions'
        assert reduce_fn in ['mean', 'sum']
        assert not self._should_call_reset, 'you should reset the state of this class after calling `finalize`'
        dof = 3
        batch_size = predictions.shape[0]
        seq_length = predictions.shape[1]

        # for simplicity we just convert angle-axis to rotation matrices
        pred_aa = np.reshape(predictions, [batch_size, seq_length, -1, dof])
        targ_aa = np.reshape(targets, [batch_size, seq_length, -1, dof])
        pred_rots = aa2rotmat(pred_aa)
        targ_rots = aa2rotmat(targ_aa)
        preds = np.reshape(pred_rots, [batch_size, seq_length, -1])
        targs = np.reshape(targ_rots, [batch_size, seq_length, -1])
        return self.compute_rotmat(preds, targs, reduce_fn)

    def compute(self, pred_motion, targets_motion, reduce_fn='mean'):
        '''
        Compute the chosen metrics. Predictions and targets can be in rotation matrix or quaternion format.
        Args:
            predictions: An np array of shape (n, seq_length, n_joints*dof)
            targets: An np array of the same shape as `predictions`
            reduce_fn: Which reduce function to apply to the joint dimension, if applicable. Choices are [mean, sum].

        Returns:
            A dictionary {metric_name -> values} where the values are given per batch entry and frame as an np array
            of shape (n, seq_length). `reduce_fn` is only applied to metrics where it makes sense
            and euler angle differences.
        '''
        if self.rep == 'rotmat':
            return self.compute_rotmat(pred_motion, targets_motion, reduce_fn)
        elif self.rep == 'quat':
            return self.compute_quat(pred_motion, targets_motion, reduce_fn)
        else:
            return self.compute_aa(pred_motion, targets_motion, reduce_fn)

    def aggregate(self, new_metrics):
        '''
        Aggregate the metrics.
        Args:
            new_metrics: Dictionary of new metric values to aggregate. Each entry is expected to be a numpy array
            of shape (batch_size, seq_length).
        '''
        assert isinstance(new_metrics, dict)
        assert list(new_metrics.keys()) == list(self.metrics_agg.keys())

        # sum over the batch dimension
        for m in new_metrics:
            if self.metrics_agg[m] is None:
                self.metrics_agg[m] = np.sum(new_metrics[m], axis=0)
            else:
                self.metrics_agg[m] += np.sum(new_metrics[m], axis=0)

        # keep track of the total number of samples processed
        batch_size = new_metrics[list(new_metrics.keys())[0]].shape[0]
        self.n_samples += batch_size

    def compute_and_aggregate(self, pred_motion, targets_motion, reduce_fn='mean'):
        '''
        Computes the metric values and aggregates them directly.
        Args:
            predictions: An np array of shape (n, seq_length, n_joints*dof)
            targets: An np array of the same shape as `predictions`
            reduce_fn: Which reduce function to apply to the joint dimension, if applicable. Choices are [mean, sum].
        '''
        new_metrics = self.compute(pred_motion, targets_motion, reduce_fn)
        self.aggregate(new_metrics)

    def get_final_metrics(self):
        '''
        Finalize and return the metrics - this should only be called once all the data has been processed.
        Returns:
            A dictionary of the final aggregated metrics per time step.
        '''
        self._should_call_reset = True  # make sure to call `reset` before new values are computed
        assert self.n_samples > 0

        for m in self.metrics_agg:
            self.metrics_agg[m] = self.metrics_agg[m] / self.n_samples

        # return a copy of the metrics so that the class can be re-used again immediately
        return copy.deepcopy(self.metrics_agg)

    @classmethod
    def get_summary_string(cls, metric_results, at_mode=False):
        '''
        Create a summary string (e.g. for printing to the console) from the given metrics for the entire sequence.
        Args:
            metric_results: Dictionary of metric values, expects them to be in shape (seq_length, ).
            at_mode: If true will report the numbers at the last frame rather then until the last frame.

        Returns:
            A summary string.
        '''
        seq_length = metric_results[list(metric_results.keys())[0]].shape[0]
        s = 'metrics until {}:'.format(seq_length)
        for m in sorted(metric_results):
            val = metric_results[m][seq_length - 1] if at_mode else np.sum(metric_results[m])
            s += '   {}: {:.3f}'.format(m, val)

        return s

    @classmethod
    def get_summary_string_all(cls, metric_results, target_lengths, at_mode=False, tb_writer=None, step=0, training=False, train_loss=None, val_loss=None):
        '''
        Create a summary string for given lengths. 
        Args:
            metric_results: Dictionary of metric values, expects them to be in shape (seq_length, ).
            target_lengths: Metrics at these time-steps are reported.
            at_mode: If true will report the numbers at the last frame rather then until the last frame.
            tb_writer: Summary writer for reporting on tensorboard
            step: Epoch number
            training: If true, train loss and val loss are reported 
            train_loss: Training loss
            val_loss: Validation loss 

        Returns:
            A summary string, and results are shown on tensorboard if it is given.
        '''
        s = ''
        for seq_length in sorted(target_lengths):
            if at_mode:
                s += '\nat frame {:<2}:'.format(seq_length)
            else:
                s += '\nMetrics until {:<2}:'.format(seq_length)
            tbs = 'until {:<2}/'.format(seq_length)
            for m in sorted(metric_results):
                val = metric_results[m][seq_length - 1] if at_mode else np.sum(metric_results[m][:seq_length])
                s += '   {}: {:.5f}'.format(m, val)
        
                if tb_writer:
                    tb_writer.add_scalar(tbs + m, val, step)
            if training:
                if tb_writer:
                    tb_writer.add_scalar('train_loss', train_loss, step)
                    tb_writer.add_scalar('val_loss', val_loss, step)
            
        return s


##############################################
def aa2rotmat(angle_axes):
    '''
    Convert angle-axis to rotation matrices using opencv's Rodrigues formula.
    Args:
        angle_axes: A np array of shape (..., 3)

    Returns:
        A np array of shape (..., 3, 3)
    '''
    orig_shape = angle_axes.shape[:-1]
    aas = np.reshape(angle_axes, [-1, 3])
    rots = np.zeros([aas.shape[0], 3, 3])
    for i in range(aas.shape[0]):
        rots[i] = cv2.Rodrigues(aas[i])[0]
    return np.reshape(rots, orig_shape + (3, 3))


##############################################
def rotmat2aa(rotmats):
    """
    Convert rotation matrices to angle-axis using opencv's Rodrigues formula.
    Args:
        rotmats: A np array of shape (..., 3, 3)

    Returns:
        A np array of shape (..., 3)
    """
    assert rotmats.shape[-1] == 3 and rotmats.shape[-2] == 3 and len(rotmats.shape) >= 3, 'invalid input dimension'
    orig_shape = rotmats.shape[:-2]
    rots = np.reshape(rotmats, [-1, 3, 3])
    aas = np.zeros([rots.shape[0], 3])
    for i in range(rots.shape[0]):
        aas[i] = np.squeeze(cv2.Rodrigues(rots[i])[0])
    return np.reshape(aas, orig_shape + (3,))


##############################################
def eye(n, batch_shape):
    iden = np.zeros(np.concatenate([batch_shape, [n, n]]))
    iden[..., 0, 0] = 1.0
    iden[..., 1, 1] = 1.0
    iden[..., 2, 2] = 1.0
    return iden