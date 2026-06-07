
# Common Libraries
import mink
import mujoco
import numpy as np
import pandas as pd
import xarray as xr
import mediapy as media
import xml.etree.ElementTree as ET
from scipy.optimize import minimize
from mink import Configuration, SE3, FrameTask, solve_ik
from scipy.spatial.transform import Rotation as R

# Custom Libraries
from sj_datastructure import make_3d_dataset

# Functions
def align_joint_link(source_ds, parent_label, child_label, target_vec, affected_labels=None):
    """
    Aligns the direction of a specific link (Parent -> Child) to a target vector 
     and rotates all associated descendant joints accordingly.

    :param source_ds: The input xarray Dataset containing 'Labels' and '3D' coordinates.
    :param parent_label: The label of the joint acting as the pivot (e.g., "S" for Shoulder).
    :param child_label: The label of the joint used to define the current direction (e.g., "E" for Elbow).
    :param target_vec: The desired 3D direction vector [x, y, z] to align the link toward.
    :param affected_labels: A list of joint labels to be rotated. If None, all joints except the parent are rotated.
    :return: A new xarray Dataset with the rotation applied, leaving the original dataset unchanged.
    """
    
    # 1. Create a deep copy to ensure the original dataset is not modified (Memory isolation)
    new_ds = source_ds.copy(deep=True)
    
    # 2. Extract joint positions
    try:
        parent_pos = new_ds.sel(Labels=parent_label)["3D"].to_numpy()[0]
        child_pos = new_ds.sel(Labels=child_label)["3D"].to_numpy()[0]
    except KeyError as e:
        print(f"Error: Label {e} not found in the dataset.")
        return new_ds

    # 3. Calculate and normalize current and target vectors
    current_vec = child_pos - parent_pos
    curr_norm = np.linalg.norm(current_vec)
    tgt_norm = np.linalg.norm(target_vec)

    # Return if vectors are too small to avoid division by zero
    if curr_norm < 1e-6 or tgt_norm < 1e-6:
        return new_ds

    u_curr = current_vec / curr_norm
    u_tgt = np.array(target_vec) / tgt_norm

    # 4. Compute the rotation matrix that aligns u_curr to u_tgt
    try:
        # Finds the optimal rotation (Quaternion) between two vectors
        rotation, _ = R.align_vectors([u_tgt], [u_curr])
    except ValueError:
        # Handles cases where vectors might be exactly opposite or invalid
        return new_ds

    # 5. Define joints to be rotated
    # If no list is provided, assume all joints except the pivot (parent) should move
    if affected_labels is None:
        all_labels = new_ds["Labels"].values.tolist()
        affected_labels = [l for l in all_labels if l != parent_label]

    R_mat = rotation.as_matrix()
    
    # 6. Apply rotation to each affected joint relative to the parent position
    for label in affected_labels:
        if label not in new_ds["Labels"].values:
            continue
            
        # Get original coordinates
        original_pos = new_ds.sel(Labels=label)["3D"].values[0]
        
        # [Core Logic] 
        # 1. Translate joint so the parent is at the origin
        translated = original_pos - parent_pos
        # 2. Apply 3D rotation
        rotated = rotation.apply(translated)
        # 3. Translate joint back to the parent's actual world position
        final_pos = rotated + parent_pos
        
        # Update the value in the new dataset
        new_ds["3D"].loc[dict(Labels=label)] = final_pos[np.newaxis, :]

    return new_ds, R_mat

def do_IK(configuration: mink.Configuration, 
          tasks: list,
          renderer: mujoco.Renderer, 
          n_frames: int,
          dt: float,
          target_info: dict,
          is_render = True,
          camera: mujoco.MjvCamera = None,
          scene_option: mujoco.MjvOption = None,
          solver_name: str = "quadprog",
          limits: list = [],
          rendering_fps: int = 60,
          is_clear_pos: bool = True,
          static_pos_name: str = "home"):
    """
    Do inverse kinematics with mujoco

    :param configuration:
    :param tasks: A list of mink tasks (e.g., FrameTask, EndEffectorTask) to satisfy.
    :param renderer: MuJoCo renderer for visual feedback.
    :param n_frames: Total number of frames to simulate.
    :param dt: Time step for the IK integration
    :param target_info: the target position for aligning model's site position. The key represents name of site and value need to be set as time series data
    :param is_render: Whether to capture and display video frames
    :param scene_option: MuJoCo visualization options (e.g., to toggle frames or transparency)
    :param solver_name: The optimization solver for the IK (default: "quadprog")
    :param limits: Constraints for the solver (e.g., joint limits)
    :param rendering_fps: Frames per second for the output video
    :param is_clear_pos: If True, resets the model to a keyframe before starting
    :param static_pos_name: The name of the keyframe to reset to
    """
    if scene_option is None:
        scene_option = mujoco.MjvOption()

    if camera is None:
        camera = mujoco.MjvCamera()
    
    if is_clear_pos:
        configuration.update_from_keyframe(static_pos_name)

    task_target_map = []
    for task in tasks:
        if isinstance(task, FrameTask) and task.frame_name in target_info:
            task_target_map.append((task, target_info[task.frame_name]))
    
    frames = []
    for frame_i in range(n_frames):
        for task, target_trajectory in task_target_map:
            target_val = target_trajectory[frame_i]
            new_target = SE3(wxyz_xyz=target_val)
            task.set_target(new_target)

            mocap_id = getattr(task, 'mocap_id', None) 
            if mocap_id is not None:
                configuration.data.mocap_pos[mocap_id] = new_target.wxyz_xyz[4:]
                configuration.data.mocap_quat[mocap_id] = new_target.wxyz_xyz[:4]
            
        # Perform inverse kinematics
        vel = solve_ik(configuration, tasks, dt, solver=solver_name, limits = limits)
        configuration.integrate_inplace(vel, dt)
        
        # Update the physical state of the simulation
        mujoco.mj_forward(configuration.model, configuration.data)
        
        # Save the current frame for video
        renderer.update_scene(configuration.data, camera = camera, scene_option = scene_option)
            
        if is_render:
            pixels = renderer.render()
            frames.append(pixels)

    if is_render:
        media.show_video(frames, fps = rendering_fps, loop = False)

def transform_ds(df, name_col):
    names = df[name_col].to_numpy()
    locs = df[["global_x", "global_y", "global_z"]].to_numpy()
    
    ds = make_3d_dataset(locs[None, :, :],
                         "3D",
                         element_dataset_names = ["Times", "Labels", "Coords"],
                         dataset1_dim_names = np.arange(1),
                         dataset2_dim_names = names,
                         dataset3_dim_names = ["X", "Y", "Z"])
    return ds
