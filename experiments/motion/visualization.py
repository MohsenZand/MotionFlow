import os
import subprocess
import numpy as np
import cv2
import quaternion
from colorama import Fore, Style
from matplotlib import pyplot as plt
from matplotlib import animation as animation

from utils import get_closest_rotmat, is_valid_rotmat, sparse_to_full
from metrics import aa2rotmat

CMU_JOINTS_TO_IGNORE = [6, 7, 12, 19, 20, 24, 25, 27, 28, 32, 33, 35, 36, 37]
CMU_MAJOR_JOINTS = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 13,
                    14, 15, 16, 17, 18, 21, 22, 23, 26, 29, 30, 31, 34]
CMU_NR_JOINTS = 38
CMU_PARENTS = np.array([0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14, 15, 16, 17,
                       18, 19, 16, 21, 22, 23, 24, 25, 26, 24, 28, 16, 30, 31, 32, 33, 34, 35, 33, 37]) - 1

_prop_cycle = plt.rcParams['axes.prop_cycle']
_colors = _prop_cycle.by_key()['color']


################################
class ForwardKinematics(object):
    """
    https://github.com/eth-ait/spl
    FK Engine.
    """

    def __init__(self, offsets, parents, left_mult=False, major_joints=None, norm_idx=None, no_root=True):
        self.offsets = offsets
        if norm_idx is not None:
            self.offsets = self.offsets / \
                np.linalg.norm(self.offsets[norm_idx])
        self.parents = parents
        self.n_joints = len(parents)
        self.major_joints = major_joints
        self.left_mult = left_mult
        self.no_root = no_root
        assert self.offsets.shape[0] == self.n_joints

    def fk(self, joint_angles):
        """
        Perform forward kinematics. This requires joint angles to be in rotation matrix format.
        Args:
            joint_angles: np array of shape (N, n_joints*3*3)

        Returns:
            The 3D joint positions as an array of shape (N, n_joints, 3)
        """
        assert joint_angles.shape[-1] == self.n_joints * 9
        angles = np.reshape(joint_angles, [-1, self.n_joints, 3, 3])
        n_frames = angles.shape[0]
        positions = np.zeros([n_frames, self.n_joints, 3])
        # intermediate storage of global rotation matrices
        rotations = np.zeros([n_frames, self.n_joints, 3, 3])
        if self.left_mult:
            offsets = self.offsets[np.newaxis,
                                   np.newaxis, ...]  # (1, 1, n_joints, 3)
        else:
            offsets = self.offsets[np.newaxis, ...,
                                   np.newaxis]  # (1, n_joints, 3, 1)

        if self.no_root:
            angles[:, 0] = np.eye(3)

        for j in range(self.n_joints):
            if self.parents[j] == -1:
                # this is the root, we don't consider any root translation
                positions[:, j] = 0.0
                rotations[:, j] = angles[:, j]
            else:
                # this is a regular joint
                if self.left_mult:
                    positions[:, j] = np.squeeze(np.matmul(
                        offsets[:, :, j], rotations[:, self.parents[j]])) + positions[:, self.parents[j]]
                    rotations[:, j] = np.matmul(
                        angles[:, j], rotations[:, self.parents[j]])
                else:
                    positions[:, j] = np.squeeze(np.matmul(
                        rotations[:, self.parents[j]], offsets[:, j])) + positions[:, self.parents[j]]
                    rotations[:, j] = np.matmul(
                        rotations[:, self.parents[j]], angles[:, j])

        return positions

    def from_aa(self, joint_angles):
        """
        Get joint positions from angle axis representations in shape (N, n_joints*3).
        """
        angles = np.reshape(joint_angles, [-1, self.n_joints, 3])
        angles_rot = np.zeros(angles.shape + (3,))
        for i in range(angles.shape[0]):
            for j in range(self.n_joints):
                angles_rot[i, j] = cv2.Rodrigues(angles[i, j])[0]
        return self.fk(np.reshape(angles_rot, [-1, self.n_joints * 9]))

    def from_rotmat(self, joint_angles):
        """
        Get joint positions from rotation matrix representations in shape (N, H36M_NR_JOINTS*3*3).
        """
        return self.fk(joint_angles)
        
    def from_sparse(self, joint_angles_sparse, rep="rotmat", return_sparse=True):
        """
        Get joint positions from reduced set of H36M joints.
        Args:
            joint_angles_sparse: np array of shape (N, len(sparse_joint_idxs) * dof))
            sparse_joints_idxs: List of indices into `H36M_JOINTS` pointing out which SMPL joints are used in
              `pose_sparse`. If None defaults to `H36M_MAJOR_JOINTS`.
            rep: "rotmat" or "quat", which representation is used for the angles in `joint_angles_sparse`
            return_sparse: If True it will return only the positions of the joints given in `sparse_joint_idxs`.

        Returns:
            The joint positions as an array of shape (N, len(sparse_joint_idxs), 3) if `return_sparse` is True
            otherwise (N, H36M_NR_JOINTS, 3).
        """
        assert self.major_joints is not None
        assert rep in ["rotmat", "quat", "aa"]
        joint_angles_full = sparse_to_full(joint_angles_sparse, self.major_joints, self.n_joints, rep)
        fk_func = self.from_quat if rep == "quat" else self.from_aa if rep == "aa" else self.from_rotmat
        positions = fk_func(joint_angles_full)
        if return_sparse:
            positions = positions[:, self.major_joints]
        return positions


################################
class CMUForwardKinematics(ForwardKinematics):
    """
    Forward Kinematics for the skeleton defined by CMU dataset.
    """

    def __init__(self):
        offset = 70 * np.array([0, 0, 0, 0, 0, 0, 1.65674000000000, -1.8028200000000000,
                                0.624770000000000, 2.59720000000000, -7.135760000000000,
                                0, 2.49236000000000, -6.84770000000000, 0, 0.1970400000,
                                -0.5413600000000000, 2.145810000000000, 0.00000, 0.0000,
                                1.11249000000000, 0.000, 0.000, 0.00, -1.61070000000000,
                                -1.80282000000000, 0.624760000000000, -2.59502000000000,
                                -7.1297700000000, 0, -2.4678000000000, -6.7802400000000,
                                0, -0.230240000000000, -0.63258000000000, 2.13368000000,
                                0, 0, 1.11569000000000, 0.0, 0.0, 0, 0.0196100000000000,
                                2.05450000000000, -0.141120000000000, 0.010210000000000,
                                2.06436000000000, -0.0592100000000000, 0.00, 0.00, 0.00,
                                0.00713000000000000, 1.56711000000000, 0.14968000000000,
                                0.0342900000000000, 1.56041000000000, -0.10006000000000,
                                0.0130500000000000, 1.6256000000000, -0.052650000000000,
                                0, 0, 0, 3.54205000000, 0.90436000000, -0.1736400000000,
                                4.86513000000000, 0, 0, 3.35554000000000, 0, 0, 0, 0, 0,
                                0.661170000000000, 0, 0, 0.533060000000000, 0.0, 0.0, 0,
                                0, 0, 0.541200000000000, 0, 0.5412000000000000, 0, 0, 0,
                                -3.49802000000000, 0.75994000000000, -0.326160000000000,
                                -5.02649000000000, 0, 0, -3.364310000000, 0, 0, 0, 0, 0,
                                -0.7304100000000, 0, 0, -0.5888700000000, 0, 0, 0, 0, 0,
                                -0.597860000000000, 0, 0.597860000000000])

        offsets = offset.reshape(-1, 3)

        # normalize so that right thigh has length 1
        super(CMUForwardKinematics, self).__init__(offsets, CMU_PARENTS, norm_idx=None, left_mult=True, major_joints=CMU_MAJOR_JOINTS)


################################
class Visualizer(object):
    """
    https://github.com/eth-ait/spl
     Helper class to visualize motion. It supports an interactive mode as well as saving frames/videos.
    """

    def __init__(self, interactive, rep="rotmat", fk_engine=None,
                 output_dir=None, to_video=False, save_figs=False):
        """
        Initializer. Determines if visualizations are shown interactively or saved to disk.
        Args:
            interactive: Boolean if motion is to be shown in an interactive matplotlib window. If True, requires
              `fk_engine` and can only display skeletons because animating dense meshes is too slow. If False,
              `output_dir` must be passed. In this case, frames (and optionally a video) are dumped to disk.
              This is slow as it uses SMPL to produce meshes and joint positions for every time instance.
            rep: Representation of the input motions, 'rotmat', 'quat', or 'aa'.
            fk_engine: The forward-kinematics engine required for interactive mode.
            output_dir: Where to dump frames/videos in non-interactive mode.
            skeleton: Boolean if skeleton should be shown in non-interactive mode.
            dense: Boolean if mesh should be shown in non-interactive mode.
            to_video: Boolean if a video should be dumped to disk in non-interactive mode.
        """

        self.interactive = interactive
        self.save_figs = save_figs
        self.fk_engine = fk_engine
        self.video_dir = output_dir
        self.rep = rep
        self.to_video = to_video
        # what color to use to display ground-truth and seed
        self.base_color = _colors[2]
        # what color to use for predictions,
        self.prediction_color = _colors[3]
        self.expected_n_input_joints = len(self.fk_engine.major_joints)
        assert rep in ["rotmat", "quat", "aa"]
        if self.interactive:
            assert self.fk_engine
        else:
            assert output_dir

    def create_clip_skeleton(self, joint_angles, title):
        """Creates clip of a given sequence in rotation matrix format.

        Args:
            joint_angles: sequence of poses.
            title: output file name.
        Returns:
        """
        assert joint_angles.shape[-1] == self.expected_n_input_joints * 9
        n_joints = self.expected_n_input_joints

        # calculate positions
        joint_angles = np.reshape(joint_angles, [-1, n_joints, 3, 3])

        pos = self.fk_engine.from_sparse(joint_angles, return_sparse=False)  # (N, full_n_joints, 3)

        pos = pos[..., [0, 2, 1]]

        fname = title.replace('/', '.')
        # reduce name otherwise stupid OSes (i.e., all of them) can't handle it
        fname = fname.split('_')[0]
        dir_prefix = 'skeleton'
        out_dir = os.path.join(self.video_dir, dir_prefix, fname)

        fps = 200

        animate_matplotlib(positions=[pos], colors=[self.base_color], titles=[""], 
                            fig_title=title, parents=self.fk_engine.parents, out_dir=out_dir, 
                            fname=fname, to_video=self.to_video, fps=fps,
                            save_figs=self.save_figs, interactive=self.interactive)

    def visualize_results(self, seed, prediction, target, title):
        """
        Visualize prediction and ground truth side by side. At the moment only supports sparse pose input in rotation
        matrix or quaternion format.
        Args:
            seed: A np array of shape (seed_seq_length, n_joints*dof)
            prediction: A np array of shape (target_seq_length, n_joints*dof)
            target: A np array of shape (target_seq_length, n_joints*dof)
            title: Title of the plot
        """
        if self.rep == "quat":
            self.visualize_quat(seed, prediction, target, title)
        elif self.rep == "rotmat":
            self.visualize_rotmat(seed, prediction, target, title)
        else:
            self.visualize_aa(seed, prediction, target, title)

    def visualize_quat(self, seed, prediction, target, title):
        assert seed.shape[-1] == prediction.shape[-1] == target.shape[-1] == self.expected_n_input_joints * 4
        assert prediction.shape[0] == target.shape[0]
        dof = 4

        def _to_rotmat(x):
            b = x.shape[0]
            xq = quaternion.from_float_array(np.reshape(x, [b, -1, dof]))
            xr = quaternion.as_rotation_matrix(xq)
            return np.reshape(xr, [b, -1])

        self.visualize_rotmat(_to_rotmat(seed), _to_rotmat(prediction), _to_rotmat(target), title)

    def visualize_aa(self, seed, prediction, target, title):
        assert seed.shape[-1] == prediction.shape[-1] == target.shape[-1] == self.expected_n_input_joints * 3
        assert prediction.shape[0] == target.shape[0]
        dof = 3

        def _to_rotmat(x):
            b = x.shape[0]
            xaa = aa2rotmat(np.reshape(x, [b, -1, dof]))
            return np.reshape(xaa, [b, -1])

        self.visualize_rotmat(_to_rotmat(seed), _to_rotmat(prediction), _to_rotmat(target), title)

    def visualize_rotmat(self, seed, prediction, target, title):
        assert seed.shape[-1] == prediction.shape[-1] == target.shape[-1] == self.expected_n_input_joints * 9
        assert prediction.shape[0] == target.shape[0]
        n_joints = self.expected_n_input_joints
        dof = 9

        # stitch seed in front of prediction and target
        pred = np.concatenate([seed, prediction], axis=0)
        targ = np.concatenate([seed, target], axis=0)

        # make sure the rotations are valid
        pred_val = np.reshape(pred, [-1, n_joints, 3, 3])
        pred = get_closest_rotmat(pred_val)
        pred = np.reshape(pred, [-1, n_joints * dof])

        # check that the targets are valid
        targ_are_valid = is_valid_rotmat(np.reshape(targ, [-1, n_joints, 3, 3]))
        assert targ_are_valid, 'target rotation matrices are not valid rotations'

        # check that the targets are valid
        pred_are_valid = is_valid_rotmat(np.reshape(pred, [-1, n_joints, 3, 3]))
        assert pred_are_valid, 'predicted rotation matrices are not valid rotations'

        change_color_after_frame=seed.shape[0]
        
        # compute positions 
        pred_pos = self.fk_engine.from_sparse(pred, return_sparse=False)  # (N, full_n_joints, 3)
        targ_pos = self.fk_engine.from_sparse(targ, return_sparse=False)  # (N, full_n_joints, 3)

        # swap axes
        pred_pos = pred_pos[..., [0, 2, 1]]
        targ_pos = targ_pos[..., [0, 2, 1]]

        fname = title.replace('/', '.')
        dir_prefix = 'skeleton'
        out_dir = os.path.join(self.video_dir, dir_prefix, fname)

        fps = 50

        # Visualize predicted and target joint angles in an interactive matplotlib window, and frames and video on disk
        animate_matplotlib(positions=[pred_pos, targ_pos], colors=[self.base_color, self.base_color], 
                            titles=['prediction', 'target'], fig_title=title, 
                            parents=self.fk_engine.parents, 
                            change_color_after_frame=(change_color_after_frame, None), 
                            color_after_change=self.prediction_color, out_dir=out_dir, 
                            fname=fname, to_video=self.to_video, fps=fps,
                            save_figs=self.save_figs, interactive=self.interactive)

def animate_matplotlib(positions, colors, titles, fig_title, parents, change_color_after_frame=None, color_after_change=None, overlay=False, fps=60, out_dir=None, to_video=True, fname=None, save_figs=False, interactive=True):
    """
    Visualize motion given 3D positions. Can visualize several motions side by side. If the sequence lengths don't
    match, all animations are displayed until the shortest sequence length.
    Args:
        positions: a list of np arrays in shape (seq_length, n_joints, 3) giving the 3D positions per joint and frame
        colors: list of color for each entry in `positions`
        titles: list of titles for each entry in `positions`
        fig_title: title for the entire figure
        parents: skeleton structure
        fps: frames per second
        change_color_after_frame: after this frame id, the color of the plot is changed (for each entry in `positions`)
        color_after_change: what color to apply after `change_color_after_frame`
        overlay: if true, all entries in `positions` are plotted into the same subplot
        out_dir: output directory where the frames and video is stored. Don't pass for interactive visualization.
        to_video: whether to convert frames into video clip or not.
        fname: video file name.
    """
    seq_length = np.amin([pos.shape[0] for pos in positions])
    n_joints = positions[0].shape[1]
    pos = positions

    # create figure with as many subplots as we have skeletons
    fig = plt.figure(figsize=(16, 9))
    plt.clf()
    n_axes = 1 if overlay else len(pos)
    axes = [fig.add_subplot(1, n_axes, i + 1, projection='3d') for i in range(n_axes)]
    fig.suptitle(fig_title)

    # create point object for every bone in every skeleton
    all_lines = []
    # available_colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
    for i, joints in enumerate(pos):
        idx = 0 if overlay else i
        ax = axes[idx]
        # markersize=2.0
        lines_j = [ax.plot(joints[0:1, n, 0], joints[0:1, n, 1], joints[0:1, n, 2], '-o',
                           linewidth=4, markersize=4.0, 
                           color=colors[i])[0] for n in range(1, n_joints)]
        all_lines.append(lines_j)

        ax.set_title(titles[i])

    # dirty hack to get equal axes behaviour
    min_val = np.amin(pos[0], axis=(0, 1))
    max_val = np.amax(pos[0], axis=(0, 1))
    max_range = (max_val - min_val).max()
    Xb = 0.5 * max_range * \
        np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * \
        (max_val[0] + min_val[0])
    Yb = 0.5 * max_range * \
        np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * \
        (max_val[1] + min_val[1])
    Zb = 0.5 * max_range * \
        np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * \
        (max_val[2] + min_val[2])

    for ax in axes:
        # ax.set_aspect('equal')
        ax.axis('off')

        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

        ax.view_init(elev=0, azim=-56)

    def on_move(event):
        # find which axis triggered the event
        source_ax = None
        for i in range(len(axes)):
            if event.inaxes == axes[i]:
                source_ax = i
                break

        # transfer rotation and zoom to all other axes
        if source_ax is None:
            return

        for i in range(len(axes)):
            if i != source_ax:
                axes[i].view_init(elev=axes[source_ax].elev, azim=axes[source_ax].azim)
                axes[i].set_xlim3d(axes[source_ax].get_xlim3d())
                axes[i].set_ylim3d(axes[source_ax].get_ylim3d())
                axes[i].set_zlim3d(axes[source_ax].get_zlim3d())
        fig.canvas.draw_idle()

    c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig_text = fig.text(0.05, 0.05, '')

    def update_frame(num, positions, lines):
        for l in range(len(positions)):
            k = 0
            pos = positions[l]
            points_j = lines[l]
            for i in range(1, len(parents)):
                a = pos[num, i]
                b = pos[num, parents[i]]
                p = np.vstack([b, a])
                points_j[k].set_data(p[:, :2].T)
                points_j[k].set_3d_properties(p[:, 2].T)
                if change_color_after_frame and change_color_after_frame[l] and num >= change_color_after_frame[l]:
                    points_j[k].set_color(color_after_change)
                else:
                    points_j[k].set_color(colors[l])
                k += 1

        time_passed = '{:>.2f} seconds passed'.format(1 / 60.0 * num)
        fig_text.set_text(time_passed)

    # create the animation object, for animation to work reference to this object must be kept
    fargs = (pos, all_lines)
    line_ani = animation.FuncAnimation(fig, update_frame, seq_length, fargs=fargs, interval=1000 / fps)

    if interactive:
        plt.show()  # interactive

    if out_dir is not None:
        save_to = os.path.join(out_dir, "frames")
        if save_figs:
            if not os.path.exists(save_to):
                os.makedirs(save_to)
            # Save frames to disk.
            for j in range(0, seq_length):
                update_frame(j, *fargs)
                fig.savefig(os.path.join(save_to, 'frame_{:0>4}.{}'.format(j, "png")))
            
            print(f'{Fore.MAGENTA}Saved frames to', out_dir, f'{Style.RESET_ALL}')

        # Create a video clip.
        if to_video:
            out_file = os.path.join(out_dir, fname + ".mp4")
            save_to_movie(out_file, os.path.join(save_to, 'frame_%04d.png'), fps=fps)
            # Delete frames if they are not required to store.
            # shutil.rmtree(save_to)

    plt.close()


def save_to_movie(out_path, frame_path_format, fps=60, start_frame=0):
    """Creates an mp4 video clip by using already stored frames in png format.

    Args:
        out_path: <output-file-path>.mp4
        frame_path_format: <path-to-frames>frame_%04d.png
        fps:
        start_frame:
    Returns:
    """
    # create movie and save it to destination
    command = ['ffmpeg',
               '-start_number', str(start_frame),
               # must be this early, otherwise it is not respected
               '-framerate', str(fps),
               '-r', str(fps//2),  # output is 30 fps
               '-loglevel', 'panic',
               '-i', frame_path_format,
               '-c:v', 'libx264',
               '-preset', 'slow',
               '-profile:v', 'high',
               '-level:v', '4.0',
               '-pix_fmt', 'yuv420p',
               '-y',
               out_path]

    fnull = open(os.devnull, 'w')
    subprocess.Popen(command, stdout=fnull).wait()
    fnull.close()

    print(f'{Fore.MAGENTA}Saved video to', out_path, f'{Style.RESET_ALL}')