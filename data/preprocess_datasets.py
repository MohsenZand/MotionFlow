import numpy as np
import os
from pathlib import Path 
import pickle as pkl 
import tensorflow as tf
import cv2
import quaternion

RNG = np.random.RandomState(42)

H36M_MAJOR_JOINTS = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 24, 25, 26, 27]
H36M_NR_JOINTS = 32

CMU_MAJOR_JOINTS = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 21, 22, 23, 26, 29, 30, 31, 34]
CMU_NR_JOINTS = 38


def rotmat2quat(rotmats):
    '''
    Convert rotation matrices to quaternions. It ensures that there's no switch to the antipodal representation
    within this sequence of rotations.
    Args:
        oris: np array of shape (seq_length, n_joints*9).

    Returns: np array of shape (seq_length, n_joints*4)
    '''
    seq_length = rotmats.shape[0]
    assert rotmats.shape[1] % 9 == 0
    ori = np.reshape(rotmats, [seq_length, -1, 3, 3])
    ori_q = quaternion.as_float_array(quaternion.from_rotation_matrix(ori))
    ori_qc = correct_antipodal_quaternions(ori_q)
    ori_qc = np.reshape(ori_qc, [seq_length, -1])
    return ori_qc


def rotmat2aa(rotmats):
    '''
    Convert rotation matrices to angle-axis format.
    Args:
        oris: np array of shape (seq_length, n_joints*9).

    Returns: np array of shape (seq_length, n_joints*3)
    '''
    seq_length = rotmats.shape[0]
    assert rotmats.shape[1] % 9 == 0
    n_joints = rotmats.shape[1] // 9
    ori = np.reshape(rotmats, [seq_length*n_joints, 3, 3])
    aas = np.zeros([seq_length*n_joints, 3])
    for i in range(ori.shape[0]):
        aas[i] = np.squeeze(cv2.Rodrigues(ori[i])[0])
    return np.reshape(aas, [seq_length, n_joints*3])


def correct_antipodal_quaternions(quat):
    '''
    Removes discontinuities coming from antipodal representation of quaternions. At time step t it checks which
    representation, q or -q, is closer to time step t-1 and chooses the closest one.
    Args:
        quat: numpy array of shape (N, K, 4) where N is the number of frames and K the number of joints. K is optional,
          i.e. can be 0.

    Returns: numpy array of shape (N, K, 4) with fixed antipodal representation
    '''
    assert len(quat.shape) == 3 or len(quat.shape) == 2
    assert quat.shape[-1] == 4

    if len(quat.shape) == 2:
        quat_r = quat[:, np.newaxis].copy()
    else:
        quat_r = quat.copy()

    def dist(x, y):
        return np.sqrt(np.sum((x - y) ** 2, axis=-1))

    # Naive implementation looping over all time steps sequentially.
    # For a faster implementation check the QuaterNet paper.
    quat_corrected = np.zeros_like(quat_r)
    quat_corrected[0] = quat_r[0]
    for t in range(1, quat.shape[0]):
        diff_to_plus = dist(quat_r[t], quat_corrected[t - 1])
        diff_to_neg = dist(-quat_r[t], quat_corrected[t - 1])

        # diffs are vectors
        qc = quat_r[t]
        swap_idx = np.where(diff_to_neg < diff_to_plus)
        qc[swap_idx] = -quat_r[t, swap_idx]
        quat_corrected[t] = qc
    quat_corrected = np.squeeze(quat_corrected)
    return quat_corrected


def to_tfexample(poses, file_id, db):
    features = dict()
    features['file_id'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[file_id.encode('utf-8')]))
    features['db_name'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[db.encode('utf-8')]))
    features['shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=poses.shape))
    features['poses'] = tf.train.Feature(float_list=tf.train.FloatList(value=poses.flatten()))
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example

def create_tfrecord_writers(output_file, n_shards):
    writers = []
    for i in range(n_shards):
        writers.append(tf.io.TFRecordWriter('{}-{:0>5d}-of-{:0>5d}'.format(output_file, i, n_shards)))
    return writers

def close_tfrecord_writers(writers):
    for w in writers:
        w.close()

def write_tfexample(writers, tf_example):
    random_writer_idx = RNG.randint(0, len(writers))
    writers[random_writer_idx].write(tf_example.SerializeToString())
    

def split_into_windows(poses, window_size, stride):
    '''Split (seq_length, dof) array into arrays of shape (window_size, dof) with the given stride.'''
    n_windows = (poses.shape[0] - window_size) // stride + 1
    windows = poses[stride * np.arange(n_windows)[:, None] + np.arange(window_size)]
    return windows


def readCSVasFloat(filename):
    '''
    Borrowed from SRNN code. Reads a csv and returns a float matrix.
    https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34

    Args
      filename: string. Path to the csv file
    Returns
      returnArray: the read data in a float32 matrix
    '''
    returnArray = []
    lines = open(filename).readlines()
    for line in lines:
        line = line.strip().split(',')
        if len(line) > 0:
            returnArray.append(np.array([np.float32(x) for x in line]))

    returnArray = np.array(returnArray)
    return returnArray


def define_actions(action, db):
    '''
    Define the list of actions we are using.

    Args
      action: String with the passed action. Could be 'all'
    Returns
      actions: List of strings of actions
    Raises
      ValueError if the action is not included in H3.6M
    '''
    if db == 'h36m':
        actions = ['walking', 'eating', 'smoking', 'discussion',  'directions',
                  'greeting', 'phoning', 'posing', 'purchases', 'sitting',
                  'sittingdown', 'takingphoto', 'waiting', 'walkingdog',
                  'walkingtogether']

        if action in actions:
            return [action]

        if action == 'all':
            return actions

        if action == 'all_srnn':
            return ['walking', 'eating', 'smoking', 'discussion']

        raise(ValueError, 'Unrecognized action: %d' % action)

    elif db == 'cmu':
        actions = ['basketball', 'basketball_signal', 'directing_traffic', 'jumping', 
                    'running', 'soccer', 'walking', 'washwindow']
        if action in actions:
            return [action]

        if action == 'all':
            return actions

        raise (ValueError, 'Unrecognized action: %d' % action)


def process_split(db, poses, file_ids, output_path, n_tfs, compute_stats, create_windows=None):
    print('storing into {} computing stats {}'.format(output_path, 'YES' if compute_stats else 'NO'))

    assert db in ['amass', 'h36m', 'cmu'], 'Unknown dataset! db must be amass, h36m, or cmu'
    
    if compute_stats:
        assert create_windows is None, 'computing the statistics should only be done when not extracting windows'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # save data as tfrecords
    tfrecord_writers = create_tfrecord_writers(os.path.join(output_path, db), n_tfs)
    tfrecord_writers_dyn = None
    if create_windows is not None:
        if not os.path.exists(output_path + '_dynamic'):
            os.makedirs(output_path + '_dynamic')
        tfrecord_writers_dyn = create_tfrecord_writers(os.path.join(output_path + '_dynamic', db), n_tfs)

    # compute normalization stats online
    n_all, mean_all, var_all, m2_all = 0.0, 0.0, 0.0, 0.0
    n_channel, mean_channel, var_channel, m2_channel = 0.0, 0.0, 0.0, 0.0
    min_all, max_all = np.inf, -np.inf
    min_seq_len, max_seq_len = np.inf, -np.inf

    # keep track of some stats to print in the end
    meta_stats_per_db = dict()

    for idx in range(len(poses)):
        pose = poses[idx]  # shape (seq_length, 33*3)
        assert len(pose) > 0, 'file is empty'

        if db == 'amass':
            db_name = file_ids[idx].split('/')[0]
        else:
            db_name = db

        if db_name not in meta_stats_per_db:
            meta_stats_per_db[db_name] = {'n_samples': 0, 'n_frames': 0}

        if create_windows is not None:
            if pose.shape[0] < create_windows[0]:
                continue

            # first save it without splitting into windows
            tfexample = to_tfexample(pose, '{}/{}'.format(0, file_ids[idx]), db_name)
            write_tfexample(tfrecord_writers_dyn, tfexample)

            # then split into windows and save later
            pose_w = split_into_windows(pose, create_windows[0], create_windows[1])
            assert pose_w.shape[1] == create_windows[0]

        else:
            pose_w = pose[np.newaxis, ...]

        for w in range(pose_w.shape[0]):
            poses_window = pose_w[w]
            tfexample = to_tfexample(poses_window, '{}/{}'.format(w, file_ids[idx]), db_name)
            write_tfexample(tfrecord_writers, tfexample)

            meta_stats_per_db[db_name]['n_samples'] += 1
            meta_stats_per_db[db_name]['n_frames'] += poses_window.shape[0]

            # update normalization stats
            if compute_stats:
                seq_len, feature_size = poses_window.shape

                # Global mean&variance
                n_all += seq_len * feature_size
                delta_all = poses_window - mean_all
                mean_all = mean_all + delta_all.sum() / n_all
                m2_all = m2_all + (delta_all * (poses_window - mean_all)).sum()

                # Channel-wise mean&variance
                n_channel += seq_len
                delta_channel = poses_window - mean_channel
                mean_channel = mean_channel + delta_channel.sum(axis=0) / n_channel
                m2_channel = m2_channel + (delta_channel * (poses_window - mean_channel)).sum(axis=0)

                # Global min&max values.
                min_all = np.min(poses_window) if np.min(poses_window) < min_all else min_all
                max_all = np.max(poses_window) if np.max(poses_window) > max_all else max_all

                # Min&max sequence length.
                min_seq_len = seq_len if seq_len < min_seq_len else min_seq_len
                max_seq_len = seq_len if seq_len > max_seq_len else max_seq_len

    close_tfrecord_writers(tfrecord_writers)
    if create_windows is not None:
        close_tfrecord_writers(tfrecord_writers_dyn)

    # print meta stats
    print()
    tot_samples = 0
    tot_frames = 0
    for db_i in meta_stats_per_db.keys():
        tot_frames += meta_stats_per_db[db_i]['n_frames']
        tot_samples += meta_stats_per_db[db_i]['n_samples']
        print('{:>20} -> {:>4d} sequences, {:>12d} frames'.format(db_i, meta_stats_per_db[db_i]['n_samples'], meta_stats_per_db[db_i]['n_frames']))

    print('{:>20} -> {:>4d} sequences, {:>12d} frames'.format('Total', tot_samples, tot_frames))

    # finalize and save stats
    if compute_stats:
        var_all = m2_all / (n_all - 1)
        var_channel = m2_channel / (n_channel - 1)

        # set certain std's to 1.0 like Martinez did
        var_channel[np.where(var_channel < 1e-4)] = 1.0

        stats = {'mean_all': mean_all, 'mean_channel': mean_channel, 'var_all': var_all,
                 'var_channel': var_channel, 'min_all': min_all, 'max_all': max_all,
                 'min_seq_len': min_seq_len, 'max_seq_len': max_seq_len, 'num_samples': tot_samples}

        stats_file = os.path.join(output_path, 'stats.npz')
        print('saving statistics to {} ...'.format(stats_file))
        np.savez(stats_file, stats=stats)

    return meta_stats_per_db


def load_data_h36m(path_to_dataset, subjects, actions, rep, fps_25=False):
    '''
    Borrowed and adapted from Martinez et al.

    Args
      path_to_dataset: string. directory where the data resides
      subjects: list of numbers. The subjects to load
      actions: list of string. The actions to load
      rep: Which representation to use for the data, ['aa', 'rotmat', 'quat']
    Returns
      trainData: dictionary with k:v
        k=(subject, action, subaction, 'even'), v=(nxd) un-normalized data
      completeData: nxd matrix with all the data. Used to normlization stats
    '''

    poses = []
    file_ids = []

    for subj in subjects:
        for action_idx in np.arange(len(actions)):
            action = actions[action_idx]
            for subact in [1, 2]:  # subactions
                print('Reading subject {0}, action {1}, subaction {2}'.format(subj, action, subact))
                filename = '{0}/S{1}/{2}_{3}.txt'.format(path_to_dataset, subj, action, subact)
                action_sequence = readCSVasFloat(filename)

                # remove the first three dimensions (root position) and the unwanted joints
                action_sequence = action_sequence[:, 3:]
                action_sequence = np.reshape(action_sequence, [-1, H36M_NR_JOINTS, 3])
                action_sequence = action_sequence[:, H36M_MAJOR_JOINTS]
                action_sequence = np.reshape(action_sequence, [-1, len(H36M_MAJOR_JOINTS) * 3])

                n_samples, dof = action_sequence.shape
                n_joints = dof // 3

                if rep == 'rotmat':
                    expmap = np.reshape(action_sequence, [n_samples*n_joints, 3])
                    # first three values are positions, so technically it's meaningless to convert them,
                    # but we do it anyway because later we discard this values 
                    rotmats = np.zeros([n_samples*n_joints, 3, 3])
                    for i in range(rotmats.shape[0]):
                        rotmats[i] = cv2.Rodrigues(expmap[i])[0]
                    action_sequence = np.reshape(rotmats, [n_samples, n_joints*3*3])
                elif rep == 'quat':
                    expmap = np.reshape(action_sequence, [n_samples * n_joints, 3])
                    quats = quaternion.from_rotation_vector(expmap)
                    action_sequence = np.reshape(quaternion.as_float_array(quats), [n_samples, n_joints*4])
                else:
                    pass  # the data is already in angle-axis format


                # downsample to 25 fps
                if fps_25:
                    even_list = range(0, n_samples, 2)
                else:
                    even_list = range(0, n_samples, 1)
                    
                poses.append(action_sequence[even_list, :])
                file_ids.append('S{}_{}_{}'.format(subj, action, subact))

    return poses, file_ids


def load_data_cmu(path_to_dataset, mode, actions, rep, fps_25=False):
    '''
    Borrowed from SRNN code. This is how the SRNN code reads the provided .txt files
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L270

    Args
        path_to_dataset: string. directory where the data resides
        subjects: list of numbers. The subjects to load
        actions: list of string. The actions to load
    Returns
        trainData: dictionary with k:v
        k=(subject, action, subaction, 'even'), v=(nxd) un-normalized data
        completeData: nxd matrix with all the data. Used to normlization stats
    '''

    poses = []
    file_ids = []
    data_dir = Path(path_to_dataset) / mode
    
    for action_idx in range(len(actions)):
        action = actions[action_idx]
        path = '{}/{}'.format(data_dir, action)
        n_files = len(os.listdir(path))
        for i in range(n_files):
            print('Reading action {0} {1}'.format(action, i+1))
            filename = '{}/{}/{}_{}.txt'.format(data_dir, action, action, i+1)
            action_sequence = readCSVasFloat(filename)

            # remove the first three dimensions (root position) and the unwanted joints
            action_sequence = action_sequence[:, 3:]
            action_sequence = np.reshape(action_sequence, [-1, CMU_NR_JOINTS, 3])
            action_sequence = action_sequence[:, CMU_MAJOR_JOINTS]
            action_sequence = np.reshape(action_sequence, [-1, len(CMU_MAJOR_JOINTS) * 3])

            n_samples, dof = action_sequence.shape
            n_joints = dof // 3

            if rep == 'rotmat':
                expmap = np.reshape(action_sequence, [n_samples*n_joints, 3])
                # first three values are positions, so technically it's meaningless to convert them,
                # but we do it anyway because later we discard this values 
                rotmats = np.zeros([n_samples*n_joints, 3, 3])
                for i in range(rotmats.shape[0]):
                    rotmats[i] = cv2.Rodrigues(expmap[i])[0]
                action_sequence = np.reshape(rotmats, [n_samples, n_joints*3*3])
            elif rep == 'quat':
                expmap = np.reshape(action_sequence, [n_samples * n_joints, 3])
                quats = quaternion.from_rotation_vector(expmap)
                action_sequence = np.reshape(quaternion.as_float_array(quats), [n_samples, n_joints*4])
            else:
                pass  # the data is already in angle-axis format


            # downsample to 25 fps
            if fps_25:
                even_list = range(0, n_samples, 2)
            else:
                even_list = range(0, n_samples, 1)
                
            poses.append(action_sequence[even_list, :])
            file_ids.append('{}'.format(action))

    return poses, file_ids


def read_fnames(from_):
    with open(from_, 'r') as fh:
        lines = fh.readlines()
        return [line.strip() for line in lines]

def find_files_amass(RAW_data_dir, data_dir):
    train_fnames = read_fnames(os.path.join(data_dir, 'training_fnames.txt'))
    valid_fnames = read_fnames(os.path.join(data_dir, 'validation_fnames.txt'))
    test_fnames = read_fnames(os.path.join(data_dir, 'test_fnames.txt'))
    
    # Load all available filenames from the source directory.
    train_fnames_avail = []
    test_fnames_avail = []
    valid_fnames_avail = []
    for root_dir, dir_names, file_names in os.walk(RAW_data_dir):
        dir_names.sort()
        for f in sorted(file_names):
            if f.endswith('.pkl'):
                # Extract name of the database.
                db_name = os.path.split(os.path.dirname(os.path.join(root_dir, f)))[1]
                db_name = '_'.join(db_name.split('_')[1:]) if 'AMASS' in db_name else db_name.split('_')[0]
                file_id = '{}/{}'.format(db_name, f)

                if file_id in train_fnames:
                    train_fnames_avail.append((root_dir, f, file_id))
                elif file_id in valid_fnames:
                    valid_fnames_avail.append((root_dir, f, file_id))
                elif file_id in test_fnames:
                    test_fnames_avail.append((root_dir, f, file_id))
                else:
                    # This file was rejected because its sequence length is smaller than 180 (3 seconds)
                    pass

    tot_files = len(train_fnames_avail) + len(test_fnames_avail) + len(valid_fnames_avail)
    print('found {} training files {:.2f} %'.format(len(train_fnames_avail), len(train_fnames_avail) / tot_files * 100.0))
    print('found {} validation files {:.2f} %'.format(len(valid_fnames_avail), len(valid_fnames_avail) / tot_files * 100.0))
    print('found {} test files {:.2f} %'.format(len(test_fnames_avail), len(test_fnames_avail) / tot_files * 100.0))

    return train_fnames_avail, valid_fnames_avail, test_fnames_avail


def load_data_amass(fnames, fps_30=False):

    poses = []
    file_ids = []

    for idx in range(len(fnames)):
        root_dir, f, file_id = fnames[idx]
        with open(os.path.join(root_dir, f), 'rb') as f_handle:
            print('\r [{:0>5d} / {:0>5d}] processing file {}'.format(idx + 1, len(fnames), f), end='')
            data = pkl.load(f_handle, encoding='latin1')
            seq = np.array(data['poses'])  # shape (seq_length, 135)
            assert len(seq) > 0, 'file is empty'

            if rep == 'quat':
                # convert to quaternions
                seq = rotmat2quat(seq)
            elif rep == 'aa':
                seq = rotmat2aa(seq)
            else:
                pass

            n_samples, dof = seq.shape
            
            # downsample to 30 fps
            if fps_30:
                even_list = range(0, n_samples, 2)
            else:
                even_list = range(0, n_samples, 1)
                
            poses.append(seq[even_list, :])
            file_ids.append('{}'.format(file_id))


    return poses, file_ids



if __name__ == '__main__':
    
    reps = ['aa', 'rotmat', 'quat']
    dbs = ['amass', 'h36m', 'cmu']

    for db in dbs:
        data_dir = Path(<PATH TO DATASETS>) / db
        n_tfs = 20 if db == 'amass' else 5  # '# of tfrecord files to create per split'
        for rep in reps:
            if db == 'h36m':
                window_size = 150  # 3 seconds
                window_stride = 10  
                train_subjects = [1, 6, 7, 8, 9, 11]  # for h3.6m this is fixed
                test_subjects = [5]  # for h3.6m this is fixed, use test subject as validation
                actions = define_actions('all', 'h36m')
                train_data, train_ids = load_data_h36m(data_dir, train_subjects, actions, rep=rep, fps_25=False)
                test_data, test_ids = load_data_h36m(data_dir, test_subjects, actions, rep=rep, fps_25=False)
                valid_data, valid_ids = test_data, test_ids

            elif db == 'cmu':
                window_size = 150  # 3 seconds
                window_stride = 10 
                actions = define_actions('all', 'h36m')
                train_data, train_ids = load_data_cmu(data_dir, 'train', actions, rep=rep, fps_25=False)
                test_data, test_ids = load_data_cmu(data_dir, 'test', actions, rep=rep, fps_25=False)
                valid_data, valid_ids = test_data, test_ids

            elif db == 'amass':
                RAW_AMASS_dir = <PATH TO RAW AMASS DATASET>
                window_size = 180  # 3 seconds
                window_stride = 30  
                train_fnames_avail, valid_fnames_avail, test_fnames_avail = find_files_amass(RAW_AMASS_dir, data_dir)   
                train_data, train_ids = load_data_amass(train_fnames_avail, fps_30=False)
                valid_data, valid_ids = load_data_amass(valid_fnames_avail, fps_30=False)
                test_data, test_ids = load_data_amass(test_fnames_avail, fps_30=False)


            print('process training data ...')
            tr_stats = process_split(db, train_data, train_ids, os.path.join(data_dir, rep, 'training'), n_tfs, compute_stats=True, create_windows=None)
            tr_stats = process_split(db, train_data, train_ids, os.path.join(data_dir, rep, 'training'), n_tfs, compute_stats=False, create_windows=(window_size, window_stride))

            print('process validation data ...')
            va_stats = process_split(db, valid_data, valid_ids, os.path.join(data_dir, rep, 'validation'), n_tfs, compute_stats=False, create_windows=(window_size, window_stride))

            print('process test data ...')
            te_stats = process_split(db, test_data, test_ids, os.path.join(data_dir, rep, 'test'), n_tfs, compute_stats=False, create_windows=(window_size, window_stride))

            print('Meta stats for all splits combined')
            total_stats = tr_stats
            for db in tr_stats.keys():
                for k in tr_stats[db].keys():
                    total_stats[db][k] += va_stats[db][k] if db in va_stats else 0
                    total_stats[db][k] += te_stats[db][k] if db in te_stats else 0

            tot_samples = 0
            tot_frames = 0
            for db in total_stats.keys():
                tot_frames += total_stats[db]['n_frames']
                tot_samples += total_stats[db]['n_samples']
                print('{:>20} -> {:>4d} sequences, {:>12d} frames'.format(db, total_stats[db]['n_samples'], total_stats[db]['n_frames']))

        print('{:>20} -> {:>4d} sequences, {:>12d} frames'.format('Total', tot_samples, tot_frames))
