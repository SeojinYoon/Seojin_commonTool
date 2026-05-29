
# Common Libraries
import re
import mink
import mujoco
import numpy as np
import pandas as pd
import xarray as xr
import mediapy as media
from pathlib import Path
import xml.etree.ElementTree as ET
from scipy.optimize import minimize
from mink import Configuration, SE3, SO3, FrameTask, solve_ik
from scipy.spatial.transform import Rotation as R

# Custom Libraries
from sj_datastructure import make_3d_dataset
from XML.xml_util import parse_xml_with_includes

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

def get_mujoco_joints(xml_path: str) -> (list, list):
    """
    get joints from mujoco compatible model
    
    :param xml_path: xml path of mujoco model
    
    return 
        dependent_joints: joint which changes its angle depending on independent joints
        independent_joints: joint which changes its angle independently
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    dependent_joints = set()
    for eq in root.findall(".//equality/joint"):
        j1 = eq.get("joint1")
        if j1 is not None:
            dependent_joints.add(j1)

    all_joints = []
    for joint in root.findall(".//joint"):
        name = joint.get("name")
        if name is not None:
            all_joints.append(name)

    independent_joints = [
        j for j in all_joints if j not in dependent_joints
    ]

    return dependent_joints, filter_control_joints(independent_joints)

def filter_control_joints(joint_list):
    constraint_sub_fix = "_con"
    lock_sub_fix = "_locked"
    
    return [j for j in joint_list if not j.endswith(constraint_sub_fix) and not j.endswith(lock_sub_fix)]

def extract_osim_default_angles(osim_path):
    text = open(osim_path, "r", encoding="utf-8", errors="ignore").read()

    pattern = re.compile(
        r'<Coordinate name="([^"]+)">.*?<default_value>(.*?)</default_value>',
        re.DOTALL
    )

    rows = []
    for name, value in pattern.findall(text):
        value = float(value.strip())
        rows.append({
            "joint": name,
            "default_rad": value,
            "default_deg": value * 180 / 3.141592653589793
        })

    return pd.DataFrame(rows)


def extract_mujoco_keyframe_qpos(xml_path, key_name="default-pose"):
    root = ET.parse(xml_path).getroot()

    # MuJoCo qpos order follows joints inside <worldbody>
    joints = []
    for joint in root.find("worldbody").findall(".//joint"):
        name = joint.get("name")
        joint_type = joint.get("type", "hinge")

        if joint_type == "free":
            nq = 7
        elif joint_type == "ball":
            nq = 4
        else:
            nq = 1

        joints.append((name, joint_type, nq))

    key = root.find(f".//key[@name='{key_name}']")
    qpos = [float(x) for x in key.get("qpos").split()]

    rows = []
    idx = 0
    for name, joint_type, nq in joints:
        values = qpos[idx:idx+nq]

        if nq == 1:
            value = values[0]
            rows.append({
                "joint": name,
                "joint_type": joint_type,
                "qpos_index": idx,
                "default_rad": value,
                "default_deg": value * 180 / 3.141592653589793
            })
        else:
            rows.append({
                "joint_angle": name,
                "joint_type": joint_type,
                "qpos_index": idx,
                "default_rad": values,
                "default_deg": None
            })

        idx += nq

    return pd.DataFrame(rows)

def get_mujoco_joint_ranges(model_path):
    tree = ET.parse(model_path)
    root = tree.getroot()
    
    data = []
    
    for joint in root.iter('joint'):
        if 'range' in joint.attrib:
            name = joint.attrib.get('name', 'unknown')
            min_val, max_val = map(float, joint.attrib['range'].split())
            
            data.append({
                'joint_name': name,
                'min_angle': min_val,
                'max_angle': max_val
            })
    
    df = pd.DataFrame(data)
    return df

def get_opensim_joint_ranges(model_path):
    text = Path(model_path).read_text(errors="replace")
    
    start = text.find("<JointSet")
    end = text.find("</JointSet>", start) + len("</JointSet>")
    
    jointset_xml = "<root>" + text[start:end] + "</root>"
    root = ET.fromstring(jointset_xml)
    
    data = []
    
    objects = root.find(".//objects")
    
    for joint in list(objects):
        joint_name = joint.attrib.get("name")
        joint_type = joint.tag
    
        for coord in joint.findall(".//Coordinate"):
            coord_name = coord.attrib.get("name")
            range_elem = coord.find("range")
    
            if range_elem is not None and range_elem.text:
                min_val, max_val = map(float, range_elem.text.split())
    
                data.append({
                    "joint_name": joint_name,
                    "joint_type": joint_type,
                    "coordinate_name": coord_name,
                    "min_angle": min_val,
                    "max_angle": max_val
                })
    
    df = pd.DataFrame(data)
    return df

def get_opensim_independent_joints(osim_model_path):
    text = Path(osim_model_path).read_text(errors="replace")
    
    def parse_section(tag):
        start = text.find(f"<{tag}")
        end = text.find(f"</{tag}>", start) + len(f"</{tag}>")
        return ET.fromstring("<root>" + text[start:end] + "</root>")
    
    joint_root = parse_section("JointSet")
    constraint_root = parse_section("ConstraintSet")
    
    dependent_coords = set()
    
    for con in constraint_root.findall(".//CoordinateCouplerConstraint"):
        is_enforced = con.findtext("isEnforced", "").strip().lower()
        
        if is_enforced == "true":
            dep = con.findtext("dependent_coordinate_name", "").strip()
            if dep:
                dependent_coords.add(dep)
    
    data = []
    for joint in joint_root.findall(".//objects/*"):
        joint_name = joint.attrib.get("name")
        joint_type = joint.tag
    
        for coord in joint.findall(".//Coordinate"):
            coord_name = coord.attrib.get("name")
            locked = coord.findtext("locked", "").strip().lower()
    
            if locked == "false" and coord_name not in dependent_coords:
                data.append({
                    "joint_name": joint_name,
                    "joint_type": joint_type,
                    "coordinate_name": coord_name,
                    "locked": locked,
                })
    
    df_independent = pd.DataFrame(data)
    return df_independent

def get_simulated_img(model: mujoco.MjModel,
                      initial_qpos: np.ndarray,
                      qpos: np.ndarray,
                      moving_joint_names: list = None,
                      azimuth: float = -90,
                      elevation: float = -15,
                      distance: float = 2.0) -> np.ndarray:
    """
    Get simulated image

    :param model: mujoco model
    :param qpos: joint angles having the number of joint corresponding to model's joint numbers
    :param moving_joint_names: joint names to make movement if a joint name is not included in this, it does not make movement
    :param azimuth: camera param
    :param elevation: camera param
    :param distance: camera param

    return img(np.ndarray), mj_data
    """
    # Model config
    mj_data = mujoco.MjData(model)
    mj_data.qpos[:] = initial_qpos[:]
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, idx) for idx in range(model.njnt)]
        
    # Renderer
    renderer = mujoco.Renderer(model, height=400, width=600)
    scene_option = mujoco.MjvOption()
    scene_option.frame = mujoco.mjtFrame.mjFRAME_WORLD

    # Camera
    camera = mujoco.MjvCamera()
    camera.azimuth = azimuth
    camera.elevation = elevation
    camera.distance = distance

    # Set qpos
    moving_joint_names = joint_names if moving_joint_names is None else moving_joint_names
    for joint_name in moving_joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        addr = model.jnt_qposadr[jid]
        mj_data.qpos[addr] = qpos[addr]
        
    # Simulation
    mujoco.mj_fwdPosition(model, mj_data)

    # Get image
    renderer.update_scene(mj_data, camera = camera, scene_option = scene_option)
    return renderer.render(), mj_data

def get_dependent_joints(model: mujoco.MjModel,
                         target_joint_name: str) -> list[str]:
    """
    Get dependent joint names related to target_joint_name

    :param model: mujoco model
    :param target_joint_name: joint name

    return joint_names
    """
    dependent_joints = []
    
    target_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, target_joint_name)
    for i in range(model.neq):
        if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT and model.eq_obj2id[i] == target_jid:
            dep_jid = model.eq_obj1id[i]
            dep_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, dep_jid)
            dependent_joints.append(dep_name)
            
    return dependent_joints

def calc_muscle_influence_toJoint(model, qpos: np.ndarray) -> pd.DataFrame:
    """
    Calculate torque acting on each joint when muscle activates maximally

    :param model: mujoco model
    :param qpos: joint angles

    return torque when muscle acts on each joint maximally
    """
    # Initialize model & data
    data = mujoco.MjData(model)
    muscle_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator_{i}" for i in range(model.nu)]
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.act[:] = 0.0
    mujoco.mj_forward(model, data)
    
    # Muscle 
    rows = []
    for i in range(model.nu):
        row = {}
        
        act_adr = model.actuator_actadr[i]
        data.act[act_adr] = 1.0
        mujoco.mj_forward(model, data)

        for j in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
            dof = model.jnt_dofadr[j]
            row[joint_name] = data.qfrc_actuator[dof]

        data.act[act_adr] = 0.0
        rows.append(row)
    muscle_influence_df = pd.DataFrame(rows)
    muscle_influence_df.index = muscle_names
    return muscle_influence_df
    
def get_torque_range(model, qpos: np.ndarray) -> pd.DataFrame:
    """
    Calculate torque range when the model has the pose

    :param model: mujoco model
    :param qpos: joint angles

    return torque range of each joint
    """
    muscle_influence_df = calc_muscle_influence_toJoint(model, qpos)
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    
    # Return
    torque_range_df = pd.DataFrame(columns = joint_names)
    for joint_name in joint_names:
        negative_effect_mucle = muscle_influence_df[joint_name][muscle_influence_df[joint_name] < 0]
        positive_effect_mucle = muscle_influence_df[joint_name][muscle_influence_df[joint_name] > 0]
    
        torque_range_df.loc["min", joint_name] = negative_effect_mucle.sum()
        torque_range_df.loc["max", joint_name] = positive_effect_mucle.sum()
        
    return torque_range_df

def extract_mujoco_muscle_gain_params(model_path):
    model = mujoco.MjModel.from_xml_path(model_path)
    muscle_names = [model.actuator(i).name for i in range(model.nu)]
    
    actuator_gainprms = pd.DataFrame(model.actuator_gainprm)
    actuator_gainprms.index = muscle_names
    actuator_gainprms.columns = ["l_mt_min", "l_mt_max", 
                                 "F_max", "F_scale",
                                 "l_m_min", "l_m_max",
                                 "v_max",
                                 "fp_max", "fv_max",
                                 "unused"]
    return actuator_gainprms.T

def extract_mujoco_muscle_bias_params(model_path):
    model = mujoco.MjModel.from_xml_path(model_path)
    muscle_names = [model.actuator(i).name for i in range(model.nu)]
    
    actuator_gainprms = pd.DataFrame(model.actuator_biasprm)
    actuator_gainprms.index = muscle_names
    actuator_gainprms.columns = ["l_mt_min", "l_mt_max", 
                                 "F_max", "F_scale",
                                 "l_m_min", "l_m_max",
                                 "v_max",
                                 "fp_max", "fv_max",
                                 "unused"]
    return actuator_gainprms.T

def get_muscle_path(mujoco_path, muscle_name):
    tree = ET.parse(mujoco_path)
    root = tree.getroot()

    def get_body_name(elem):
        parent = {c: p for p in root.iter() for c in p}
        p = parent.get(elem)
        while p is not None:
            if p.tag == "body":
                return p.attrib.get("name")
            p = parent.get(p)
        return None
        
    sites = {}
    for s in root.iter("site"):
        name = s.attrib.get("name")
        if name:
            sites[name] = {
                "site_name": name,
                "body": get_body_name(s),
                "pos": s.attrib.get("pos"),
                "size": s.attrib.get("size"),
            }
            
    actuator = None
    for elem in root.iter():
        if elem.tag in ["muscle", "general"] and elem.attrib.get("name") == muscle_name:
            actuator = elem
            break

    tendon_name = actuator.attrib.get("tendon")
    tendon = None
    for elem in root.findall(".//tendon/*"):
        if elem.attrib.get("name") == tendon_name:
            tendon = elem
            break

    rows = []
    for i, child in enumerate(tendon):
        row = {
            "order": i,
            "element": child.tag,
            "name": child.attrib.get("site") or child.attrib.get("geom"),
            "sidesite": child.attrib.get("sidesite"),
        }

        if child.tag == "site":
            site_info = sites.get(child.attrib.get("site"), {})
            row.update(site_info)

        rows.append(row)

    return pd.DataFrame(rows)

def get_muscle_path_world(mujoco_path, muscle_name, qpos=None):
    model = mujoco.MjModel.from_xml_path(mujoco_path)
    data = mujoco.MjData(model)
    if qpos is None:
        data.qpos[:] = np.zeros_like(data.qpos)
    mujoco.mj_forward(model, data)
    
    root = parse_xml_with_includes(mujoco_path)
    actuator = None
    for elem in root.iter():
        if elem.tag in ["muscle", "general"] and elem.attrib.get("name") == muscle_name:
            actuator = elem
            break
    
    tendon_name = actuator.attrib.get("tendon")
    tendon = None
    for elem in root.findall(".//tendon/*"):
        if elem.attrib.get("name") == tendon_name:
            tendon = elem
            break
    
    rows = []
    for i, child in enumerate(tendon):
        if child.tag == "site":
            site_name = child.attrib["site"]
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            rows.append({
                "order": i,
                "element": "site",
                "site_name": site_name,
                "world_x": data.site_xpos[site_id, 0],
                "world_y": data.site_xpos[site_id, 1],
                "world_z": data.site_xpos[site_id, 2],
            })
        elif child.tag == "geom":
            continue
            rows.append({
                "order": i,
                "element": "geom",
                "geom": child.attrib.get("geom"),
                "sidesite": child.attrib.get("sidesite"),
                "note": "wrapping geom; path point is computed internally by MuJoCo"
            })
    df = pd.DataFrame(rows)
    return df

def enforce_equality_constraints(model: mujoco.MjModel,
                                 data: mujoco.MjData):
    """
    Enforce equality constraint existing in mujoco model
    
    :param model: mujoco model
    :param data: data
    """
    for i in range(model.neq):
        if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT and data.eq_active[i]:
            y_jnt = model.eq_obj1id[i]
            x_jnt = model.eq_obj2id[i]
            y_adr = model.jnt_qposadr[y_jnt]
            x_adr = model.jnt_qposadr[x_jnt]
            coef = model.eq_data[i]  # a0~a4
            x = data.qpos[x_adr]
            x0 = model.qpos0[x_adr]
            y0 = model.qpos0[y_adr]
            dx = x - x0
            data.qpos[y_adr] = (
                y0
                + coef[0]
                + coef[1] * dx
                + coef[2] * dx**2
                + coef[3] * dx**3
                + coef[4] * dx**4
            )
    mujoco.mj_forward(model, data)

def get_joint_pos(model: mujoco.MjModel, data: mujoco.MjData):
    mujoco.mj_forward(model, data)

    rows = []
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model,
                                       mujoco.mjtObj.mjOBJ_JOINT,
                                       joint_id)

        body_id = model.jnt_bodyid[joint_id]
        body_name = mujoco.mj_id2name(model,
                                      mujoco.mjtObj.mjOBJ_BODY,
                                      body_id)

        body_pos = data.xpos[body_id]
        body_rot = data.xmat[body_id].reshape(3, 3)
        local_pos = model.jnt_pos[joint_id]
        global_pos = body_pos + body_rot @ local_pos

        rows.append({
            "joint_id": joint_id,
            "joint_name": joint_name,
            "parent_body_id": body_id,
            "parent_body_name": body_name,
            "local_x": local_pos[0],
            "local_y": local_pos[1],
            "local_z": local_pos[2],
            "global_x": global_pos[0],
            "global_y": global_pos[1],
            "global_z": global_pos[2],
        })

    return pd.DataFrame(rows)

def get_body_pos(model: mujoco.MjModel, data: mujoco.MjData):
    mujoco.mj_forward(model, data)

    rows = []
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model,
                                      mujoco.mjtObj.mjOBJ_BODY,
                                      body_id)

        if body_name is None:
            continue

        global_pos = data.xpos[body_id]
        rows.append({
            "body_id": body_id,
            "body_name": body_name,
            "global_x": global_pos[0],
            "global_y": global_pos[1],
            "global_z": global_pos[2],
        })

    return pd.DataFrame(rows)

def get_site_pos(model: mujoco.MjModel, data: mujoco.MjData):
    mujoco.mj_forward(model, data)
    
    site_rows = []
    for site_id in range(model.nsite):
        site_name = mujoco.mj_id2name(model,
                                      mujoco.mjtObj.mjOBJ_SITE,
                                      site_id)
        
        parent_body_id = model.site_bodyid[site_id]
        parent_body_name = mujoco.mj_id2name(model,
                                             mujoco.mjtObj.mjOBJ_BODY,
                                             parent_body_id)

        local_pos = model.site_pos[site_id]
        global_pos = data.site_xpos[site_id]
        site_rows.append({
            "site_id": site_id,
            "site_name": site_name,
            "parent_body_id": parent_body_id,
            "parent_body_name": parent_body_name,
            "local_x": local_pos[0],
            "local_y": local_pos[1],
            "local_z": local_pos[2],
            "global_x": global_pos[0],
            "global_y": global_pos[1],
            "global_z": global_pos[2],
        })
    df_sites = pd.DataFrame(site_rows)
    return df_sites

def get_geom_pos(model, data):
    mujoco.mj_forward(model, data)

    rows = []
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model,
                                      mujoco.mjtObj.mjOBJ_GEOM,
                                      geom_id)

        body_id = model.geom_bodyid[geom_id]
        body_name = mujoco.mj_id2name(model,
                                      mujoco.mjtObj.mjOBJ_BODY,
                                      body_id)

        local_pos = model.geom_pos[geom_id]
        global_pos = data.geom_xpos[geom_id]

        rows.append({
            "geom_id": geom_id,
            "geom_name": geom_name,
            "parent_body_id": body_id,
            "parent_body_name": body_name,
            "local_x": local_pos[0],
            "local_y": local_pos[1],
            "local_z": local_pos[2],
            "global_x": global_pos[0],
            "global_y": global_pos[1],
            "global_z": global_pos[2],
        })

    return pd.DataFrame(rows)
    
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

def find_child_bodies(path, target_body_name):
    tree = ET.parse(path)
    root = tree.getroot()
    
    target_body = root.find(f".//body[@name='{target_body_name}']")
    
    child_bodies = target_body.findall(".//body")
    child_body_names = [body.get("name") for body in child_bodies]
    return child_body_names

def get_MoBL_finger_info(mobl_geom_df, fingers = ["Index", "Middle", "Ring", "Little", "Thumb"]):
    """
    Get finger information from geometry data

    :param fingers: searching finger names

    return (dictionary)
    """
    finger_info = {}
    for finger in fingers:
        if finger == "Index":
            finger_number = 2
            number_str = "Second"
        elif finger == "Middle":
            finger_number = 3
            number_str = "Third"
        elif finger == "Ring":
            finger_number = 4
            number_str = "Fourth"
        elif finger == "Little":
            finger_number = 5
            number_str = "Fifth"
        
        if finger in ["Index", "Middle", "Ring", "Little"]:
            mcp = mobl_geom_df[mobl_geom_df["geom_name"].str.contains(f"_{finger_number}rdmcp_ext|{number_str}mcp_ext")]["geom_name"].iloc[0]
            pm = mobl_geom_df[mobl_geom_df["geom_name"].str.contains(f"_{finger_number}rdpm_ext|{number_str}pm_ext")]["geom_name"].iloc[0]
            md = mobl_geom_df[mobl_geom_df["geom_name"].str.contains(f"_{finger_number}rdmd_ext|{number_str}md_ext")]["geom_name"].iloc[0]
    
            finger_info[finger] = [mcp, pm, md]
        else:
            mc = mobl_geom_df[mobl_geom_df["geom_name"].str.contains(f"hand_geom_23")]["geom_name"].iloc[0]
            mp = mobl_geom_df[mobl_geom_df["geom_name"].str.contains(f"MPthumb")]["geom_name"].iloc[0]
            ip = mobl_geom_df[mobl_geom_df["geom_name"].str.contains(f"IPthumb")]["geom_name"].iloc[0]
            finger_info[finger] = [mc, mp, ip]
    return finger_info

def get_myoarm_fingerJoint_info(myoarm_joint_df, fingers = ["Index", "Middle", "Ring", "Little", "Thumb"]):
    """
    Get finger information from geometry data

    :param fingers: searching finger names

    return (dictionary)
    """
    finger_info = {}
    for finger in fingers:
        if finger == "Index":
            finger_number = 2
        elif finger == "Middle":
            finger_number = 3
        elif finger == "Ring":
            finger_number = 4
        elif finger == "Little":
            finger_number = 5
        
        if finger in ["Index", "Middle", "Ring", "Little"]:
            mcp = myoarm_joint_df[myoarm_joint_df["joint_name"].str.contains(f"mcp{finger_number}_flexion")]["joint_name"].iloc[0]
            pm = myoarm_joint_df[myoarm_joint_df["joint_name"].str.contains(f"pm{finger_number}_flexion")]["joint_name"].iloc[0]
            md = myoarm_joint_df[myoarm_joint_df["joint_name"].str.contains(f"md{finger_number}_flexion")]["joint_name"].iloc[0]
            finger_info[finger] = [mcp, pm, md]
        else:
            mc = myoarm_joint_df[myoarm_joint_df["joint_name"].str.contains("cmc_flexion")]["joint_name"].iloc[0]
            mp = myoarm_joint_df[myoarm_joint_df["joint_name"].str.contains("mp_flexion")]["joint_name"].iloc[0]
            ip = myoarm_joint_df[myoarm_joint_df["joint_name"].str.contains("ip_flexion")]["joint_name"].iloc[0]
            finger_info[finger] = [mc, mp, ip]
    return finger_info

def calc_joint_angle(model: mujoco.MjModel,
                     data: mujoco.MjData,
                     joint_name: str):
    """
    Calculate joint angle from mujoco model

    :param model: mujoco model
    :param data: mujoco data
    :param joint_name: joint name to be searched

    :return: A tuple containing:
        - tuple (str, str): (parent_body_name, child_body_name)
        - float: rotation angle in radian
    """
    # Params
    seq = "xyz"

    # Get parent and child segment
    parent_body_name, child_body_name = get_joint_segments(joint_name)
    child_body_id = model.jnt_bodyid[joint_id]
    parent_body_id = model.body_parentid[child_body_id]

    # Get Orientation
    R_parent = data.xmat[parent_body_id].reshape(3, 3)
    R_child = data.xmat[child_body_id].reshape(3, 3)

    # Calculate angle between parenr and child
    R_relative = R_parent.T @ R_child
    matrix_trace = np.trace(R_relative)
    clipped_value = np.clip((matrix_trace - 1.0) / 2.0, -1.0, 1.0)
    pure_angle_deg = np.arccos(clipped_value)
    
    return (parent_body_name, child_body_name), pure_angle_deg

def get_joint_segments(model: mujoco.MjModel, joint_name: str) -> (str, str):
    """
    Get child and parent segment linked by joint 

    :param model: mujoco model
    :param joint_name: name of join

    :return: parent_body_name, child_body_name
    """
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

    child_body_id = model.jnt_bodyid[joint_id]
    child_body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, child_body_id)
    
    parent_body_id = model.body_parentid[child_body_id]
    parent_body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_body_id)

    return parent_body_name, child_body_name
    
def extract_geometry_info(xml_path: str) -> pd.DataFrame:
    """
    Extract geometry information from mujoco model

    :param xml_path: mujoco model path

    :return: geometry information
    """
    # Parse xml
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # mesh information: { 'mesh_name': 'file_path_or_filename' }
    mesh_dict = {}
    asset = root.find('asset')
    if asset is not None:
        for mesh in asset.findall('mesh'):
            mesh_name = mesh.get('name')
            mesh_file = mesh.get('file', 'No File Path')  # 파일 경로가 없는 경우 대비
            if mesh_name:
                mesh_dict[mesh_name] = mesh_file

    # search geometry tag
    data_list = []
    worldbody = root.find('worldbody')
    if worldbody is not None:
        for geom in worldbody.iter('geom'):
            geom_mesh = geom.get('mesh')
            geom_name = geom.get('name')

            if geom_name and geom_mesh:
                mesh_file_path = mesh_dict.get(geom_mesh, geom_mesh)
                data_list.append({
                    "geom_name": geom_name,
                    "mesh_file_path": mesh_file_path
                })
    return pd.DataFrame(data_list)
    