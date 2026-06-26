
# Common Libraries
import os, sys, mujoco, mediapy, shutil
import numpy as np
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
import ipywidgets as widgets
from IPython.display import display, clear_output
if shutil.which("nvidia-smi") is not None:
    os.environ["MUJOCO_GL"] = "egl"
import mujoco

# Custom Libraries
from XML.xml_util import parse_xml_with_includes

# KeyFrame
def extract_keyframe_qpos(xml_path, key_name = "default-pose"):
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
    
def write_keyframe(mujoco_model_path: str,
                   output_path: str,
                   key_name: str,
                   qpos: list):
    """
    Write new keyframe to mujoco model

    :param mujoco_model_path: xml path for mujoco model
    :param output_path: output path
    :param key_name: name of qpos
    :param qpos: qpos of keyframe
    """
    # XML Parsing 
    tree = ET.parse(mujoco_model_path)
    root = tree.getroot()
    
    # Qpos
    qpos = np.round(qpos, decimals=3)
    qpos_str = " ".join(map(str, qpos))
    
    # Check whether keyframe tag exists
    keyframe_tag = root.find('keyframe')
    if keyframe_tag is None:
        keyframe_tag = ET.SubElement(root, 'keyframe')
    
    # Remove existing keyframe
    for existing_key in keyframe_tag.findall('key'):
        if existing_key.get('name') == key_name:
            keyframe_tag.remove(existing_key)
            
    # Add new key element
    new_key = ET.SubElement(keyframe_tag, 'key')
    new_key.set('name', key_name)
    new_key.set('qpos', qpos_str)
    
    # Save
    ET.indent(tree, space = "  ")
    tree.write(output_path, encoding='utf-8', xml_declaration=False)
    print(f"Write {key_name} key to {output_path}")

# Body
def global_axis_to_body_frame(model, data, body_name, axis_global):
    mujoco.mj_forward(model, data)

    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name
    )

    if body_id < 0:
        raise ValueError(f"body not found: {body_name}")

    R_body_world = data.xmat[body_id].reshape(3, 3)

    axis_body = R_body_world.T @ axis_global
    axis_body = axis_body / np.linalg.norm(axis_body)

    return axis_body

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

def find_child_bodies(path, target_body_name):
    tree = ET.parse(path)
    root = tree.getroot()
    
    target_body = root.find(f".//body[@name='{target_body_name}']")
    
    child_bodies = target_body.findall(".//body")
    child_body_names = [body.get("name") for body in child_bodies]
    return child_body_names
    
# Joint
def calc_dependent_joint_angle(mjc_model: mujoco.MjModel,
                               indep_joint_names: [str],
                               joint_config: pd.Series) -> pd.DataFrame:
    """
    Calculate dependent joint angle based on joint configuration.

    :param mjc_model: MuJoCo model object.
    :param indep_joint_names: List of independent joint names to consider.
    :param joint_config: Joint configuration data with joint names as index.

    :return: DataFrame containing calculation results.
             Columns: indep_joint_i, indep_joint, indep_joint_angle, 
                      dep_joint_i, dep_joint, dep_joint_angle
    """
    # Constants
    JOINT_COUPLING = 2
    ACTIVE = 1

    n_equalities = mjc_model.neq
    joint_names = [mujoco.mj_id2name(mjc_model, mujoco.mjtObj.mjOBJ_JOINT, idx) for idx in range(mjc_model.njnt)]

    # Filter target equality
    is_joint_equalities = mjc_model.eq_type == JOINT_COUPLING
    is_active_equalities = get_equality_active(mjc_model) == ACTIVE
    
    ind_joint_inEqualities = np.array([joint_names[mjc_model.eq_obj2id[equality_i]] for equality_i in range(n_equalities)])
    dep_joint_inEqualities = np.array([joint_names[mjc_model.eq_obj1id[equality_i]] for equality_i in range(n_equalities)])
    
    is_considered = np.isin(ind_joint_inEqualities, indep_joint_names)
    is_target_equalities = is_joint_equalities & is_active_equalities & is_considered
    equalities_idx = np.where(is_target_equalities)[0]
    n_target_equalities = len(equalities_idx)
    
    rows = []
    for i, equality_i in enumerate(equalities_idx):
        # Get independent joint angle
        indep_joint = ind_joint_inEqualities[equality_i]
        indep_joint_angle = joint_config[indep_joint]
    
        # Calculate dependent joint angle based on the independent joint angle
        poly_gains = mjc_model.eq_data[equality_i][:5]
        powers = np.arange(len(poly_gains))
    
        dep_jnt = dep_joint_inEqualities[equality_i]
        dep_joint_angle = np.sum(poly_gains * (indep_joint_angle ** powers))

        # Stack result
        rows.append({
            "indep_joint_i": joint_names.index(indep_joint),
            "indep_joint": indep_joint,
            "indep_joint_angle": indep_joint_angle,
            "dep_joint_i": joint_names.index(dep_jnt),
            "dep_joint": dep_jnt,
            "dep_joint_angle": dep_joint_angle
        })

    results = pd.DataFrame(rows)
    return results
    
def get_joints(model_path: str, constraint_suffix = "_con") -> pd.DataFrame:
    """
    get joints from mujoco compatible model
    
    :param model_path: xml path of mujoco model
    :param constraint_sub_fix:
    :param lock_sub_fix: sufix for 
    return 
        dependent_joints: joint which changes its angle depending on independent joints
        independent_joints: joint which changes its angle independently
    """

    tree = ET.parse(model_path)
    root = tree.getroot()
    
    all_joint_elements = [joint for joint in root.findall(".//joint") if joint.get("name") is not None]
    all_joint_elements = [joint for joint in all_joint_elements if constraint_suffix not in joint.get("name")]
    all_joint_names = [joint.get("name") for joint in all_joint_elements]
    
    dependent_joint_elements = [joint for joint in root.findall(".//equality/joint") if joint.get("joint1") is not None]
    dependent_joint_names = [joint.get("name").replace(constraint_suffix, "") for joint in dependent_joint_elements]

    joint_df = pd.DataFrame(columns = all_joint_names)
    is_dependent = np.array([joint_name in dependent_joint_names for joint_name in all_joint_names])
    joint_df.loc["dependent"] = is_dependent
    
    is_independent = np.logical_not(is_dependent)
    joint_df.loc["independent"] = is_independent
    
    return joint_df
    
def get_joint_ranges(model_path: str) -> pd.DataFrame:
    """
    Get each joint range from mujoco model

    :param model_path: model path

    return dataframe - columns: joint, rows: min, max
    """
    tree = ET.parse(model_path)
    root = tree.getroot()
    
    joint_elements = [joint for joint in root.iter("joint") if ("range" in joint.attrib) and (type(joint) == ET.Element)]    
    joint_names = []
    data_array = np.zeros((2, len(joint_elements)))
    for i, joint_element in enumerate(joint_elements):
        name = joint_element.attrib.get("name", "unknown")
        min_val, max_val = map(float, joint_element.attrib["range"].split())
    
        data_array[0, i] = min_val
        data_array[1, i] = max_val
        joint_names.append(name)
    
    df = pd.DataFrame(data_array)
    df.columns = joint_names
    df.index = ["min", "max"]
    return df

def get_locked_joint_angle(mjc_model: mujoco.MjModel) -> pd.DataFrame:
    """
    Get locked joint angles

    :param mjc_model: mujoco model

    return joint angles(columns: ["joint_i", "joint", "angle"])
    """
    # Constants
    JOINT_COUPLING = 2
    ACTIVE = 1
    joint_names = [mujoco.mj_id2name(mjc_model, mujoco.mjtObj.mjOBJ_JOINT, idx) for idx in range(mjc_model.njnt)]
    
    # Filter target f
    eq_actives = get_equality_active(mjc_model)
    is_locked = (mjc_model.eq_type == JOINT_COUPLING) & (eq_actives == ACTIVE) & (mjc_model.eq_obj2id == -1)
    target_equalitiex_idx = np.where(is_locked)[0]

    # Get locked joints angles
    locked_joints = np.array([joint_names[mjc_model.eq_obj1id[equality_i]] for equality_i in target_equalitiex_idx])
    locked_joints_idx = np.array([mjc_model.eq_obj1id[equality_i] for equality_i in target_equalitiex_idx])
    locked_joint_angles = np.array([mjc_model.eq_data[equality_i][0] for equality_i in target_equalitiex_idx])

    df = pd.DataFrame(np.vstack([locked_joints_idx,
                                 locked_joints,
                                 locked_joint_angles]).T, columns = ["joint_i", "joint", "angle"])
    df = df.astype({
        "joint_i": np.int32,
        "joint": str,
        "angle": np.float64
    })
        
    return df
    
def get_dof_names(model: mujoco.MjModel) -> [str]:
    """
    Get degree of freedom names

    :param model: mujoco model

    return dof names
    """
    n_dof = model.nv
    dof_names = []
    for dof_i in range(n_dof):
        joint_id = model.dof_jntid[dof_i]
        joint_name = model.joint(joint_id).name
        if model.jnt_type[joint_id] in [mujoco.mjtJoint.mjJNT_BALL, mujoco.mjtJoint.mjJNT_FREE]:
            jnt_qveladr = model.jnt_dofadr[joint_id]
            dof_names.append(f"{joint_name}_{dof_i - jnt_qveladr}")
        else:
            dof_names.append(joint_name)
    return dof_names

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
        joint_axis = model.jnt_axis[joint_id]
        
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
            "joint_axis_x" : joint_axis[0],
            "joint_axis_y" : joint_axis[1],
            "joint_axis_z" : joint_axis[2],
        })

    return pd.DataFrame(rows)
    
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
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    parent_body_name, child_body_name = get_joint_segments(model, joint_name)
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
    
def global_axis_to_joint_local(model, data, joint_name, axis_global):
    mujoco.mj_forward(model, data)

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name
    )

    if joint_id < 0:
        raise ValueError(f"joint not found: {joint_name}")

    body_id = model.jnt_bodyid[joint_id]

    R_body_world = data.xmat[body_id].reshape(3, 3)

    axis_local = R_body_world.T @ axis_global
    axis_local = axis_local / np.linalg.norm(axis_local)

    return axis_local
    
def get_joint_axis_global(model, data, joint_name):
    mujoco.mj_forward(model, data)

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"joint not found: {joint_name}")

    body_id = model.jnt_bodyid[joint_id]

    axis_local = model.jnt_axis[joint_id]
    R_body_world = data.xmat[body_id].reshape(3, 3)

    axis_global = R_body_world @ axis_local
    axis_global = axis_global / np.linalg.norm(axis_global)

    return axis_global

# Site
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

def get_body_pos(model: mujoco.MjModel, data: mujoco.MjData):
    mujoco.mj_forward(model, data)

    rows = []
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        parent_id = model.body_parentid[body_id]
        parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)

        global_pos = data.xpos[body_id]
        rows.append({
            "body_id": body_id,
            "body_name": body_name,
            "parent_id": parent_id,
            "parent_name": parent_name,
            "global_x": global_pos[0],
            "global_y": global_pos[1],
            "global_z": global_pos[2],
        })

    return pd.DataFrame(rows)
    
# Geometry
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

def get_geom_pos(model_path, model, data):
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
    
    mesh_df = extract_geometry_info(model_path)
    df = pd.merge(pd.DataFrame(rows), mesh_df, on = "geom_name", how = "left")
    
    return df
    
def calc_geom_angle(model: mujoco.MjModel, 
                    data: mujoco.MjData, 
                    geom1_name: str, 
                    geom2_name: str):
    """
    Calculate angle between two geometries

    :param model: mujoco model
    :Param data: mujoco data
    :param geom1_name: geometry name1
    :param geom2_name: geometry name2

    return angle between two geometries (radian) 
    """
    geom1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom1_name)
    geom2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom2_name)

    R_1 = data.geom_xmat[geom1_id].reshape(3, 3)
    R_2 = data.geom_xmat[geom2_id].reshape(3, 3)
    R_relative = R_1.T @ R_2
    
    trace_val = np.trace(R_relative)
    cos_theta = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)

    return angle_rad
    
# Muscle
def get_muscle_length_range(model: mujoco.MjModel) -> pd.DataFrame:
    """
    Get muscle length from mujoco model

    :param model: mujoco model

    return dataframe(columns: muscles, index: min, max) 
    """
    muscle_names = [model.actuator(i).name for i in range(model.nu)]

    result = np.zeros((2, len(muscle_names)))
    for i, muscle_name in enumerate(muscle_names):
        muscle_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, muscle_name)
        length_range = model.actuator_lengthrange[muscle_act_id]
        result[0, i] = length_range[0]
        result[1, i] = length_range[1]
    result_df = pd.DataFrame(result)
    result_df.columns = muscle_names
    result_df.index = ["min", "max"]
    return result_df

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

def extract_mujoco_muscle_bias_params(model):
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

def mju_muscleBias_manually(model_path, actuator_acc0s, acutator_lengths, actuator_lengthranges):
    model = mujoco.MjModel.from_xml_path(model_path)
    muscle_bias_param_df = extract_mujoco_muscle_bias_params(model)
    
    acc0 = model.actuator_acc0
    length = mj_data.actuator_length
    length_range = model.actuator_lengthrange
    
    mt_range = muscle_bias_param_df.loc[["l_mt_min", "l_mt_max"], :].T.to_numpy()
    F_max = muscle_bias_param_df.loc["F_max", :].to_numpy()
    F_scale = muscle_bias_param_df.loc["F_scale", :].to_numpy()
    F_max = np.where(F_scale < 0, F_scale / np.maximum(mujoco.mjMINVAL, actuator_acc0s), F_max)
    l_m_max = muscle_bias_param_df.loc["l_m_max", :].to_numpy()
    fp_max = muscle_bias_param_df.loc["fp_max", :].to_numpy()
    
    # Optimum length
    l0 = (actuator_lengthranges[:, 1] - actuator_lengthranges[:, 0]) / np.maximum(mujoco.mjMINVAL, mt_range[:, 1] - mt_range[:, 0])
    
    # Normalized length
    l = mt_range[:, 0] + (acutator_lengths - actuator_lengthranges[:, 0]) / np.maximum(mujoco.mjMINVAL, l0)
    
    # half-quadratic to (L0+lmax)/2, linear beyond
    b = 0.5 * (1 + l_m_max)
    
    result1 = np.where(l <= 1, 0, np.nan)
    
    x = (l - 1) / np.maximum(mujoco.mjMINVAL, b-1)
    result2 = -F_max * fp_max * 0.5 * x * x
    result2 = np.where(l <= b, result2, result1)
    
    x = (l - b) / np.maximum(mujoco.mjMINVAL, b-1)
    result3 = -F_max * fp_max * (0.5 + x)
    result = np.where(np.isnan(result2), result3, result2)
    return result
    
def extract_muscle_gain_params(model: mujoco.MjModel) -> pd.DataFrame:
    """
    Extract muscle gain parameters

    :param model: mujoco model

    return actuator gain parameters
    """
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

def calc_muscle_influence_toJoint(model: mujoco.MjModel, qpos: np.ndarray) -> pd.DataFrame:
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
    
# Moment arm
def get_moment_arm_matrix(model: mujoco.MjModel, data: mujoco.MjData) -> pd.DataFrame:
    """
    Get momentarm from current data ste

    :param model: mujoco model
    :param data: mujoco data

    :return: moment arm dataframe (index: muscle, columns: joint)
    """

    # Constants
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    n_acutator = len(actuator_names)
    n_dof = model.nv
    
    # DOF
    dof_names = get_dof_names(model)
    
    # Update geometry
    mujoco.mj_step1(model, data)

    # Get moment arm
    AM = np.zeros((n_acutator, n_dof))
    mujoco.mju_sparse2dense(AM,
                            data.actuator_moment.reshape(-1),
                            data.moment_rownnz,
                            data.moment_rowadr,
                            data.moment_colind.reshape(-1))
    AM_df = pd.DataFrame(AM)
    AM_df.index = actuator_names
    AM_df.columns = dof_names
    return AM_df
    
# Marker
def get_marker_local_position(xml_path: str, target_marker_name: str):
    """
    Get the local relative coordinates (pos) of a specific marker (site) 
    with respect to its parent body from a MuJoCo XML file.

    :param xml_path: Path to the MuJoCo XML file
    :param target_marker_name: Name of the target marker (site) to retrieve
    
    :return: Local [x, y, z] coordinates relative to the parent body (np.ndarray)
    """
    # 1. Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 2. Find the <site> tag with the specified name anywhere in the XML tree
    # The './/site[@name=...]' syntax allows efficient global searching within the tree hierarchy.
    marker_tag = root.find(f".//site[@name='{target_marker_name}']")
    
    if marker_tag is None:
        print(f"Error: Marker '{target_marker_name}' could not be found in the XML.")
        return None
        
    # 3. Retrieve the 'pos' attribute (e.g., "0.01096 -0.06496 0.04938")
    pos_str = marker_tag.get("pos")
    
    if pos_str is None:
        print(f"Warning: 'pos' attribute is not defined for marker '{target_marker_name}'. Defaulting to [0, 0, 0].")
        return np.array([0.0, 0.0, 0.0])
        
    # 4. Parse the space-separated string into a float numpy array
    local_pos = np.fromstring(pos_str, sep=' ')
    
    return local_pos
    
# Torque
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
    
def calculate_muscle_to_joint_torque(model, mj_data, activation = 1.0) -> pd.DataFrame:
    """
    Calculate muscle contribution to joint torque

    :param model: mujoco model
    :param mj_data: mujoco data
    :param activation: muscle activation for simulation

    return joint torque when a muscle activates (columns: muscle, types, joints)
    """
    # Constants
    n_actuators = model.nu
    n_dof = model.nv
    dof_names = get_dof_names(model)
    actuator_names = np.array([model.actuator(i).name for i in range(model.nu)])
    
    # Update geometry information
    mujoco.mj_step1(model, mj_data)
    
    # Calculate active forces
    active_forces = np.zeros(n_actuators)
    passive_forces = np.zeros(n_actuators)
    for act_i in range(n_actuators):
        # Extract target muscle params
        length = mj_data.actuator_length[act_i]
        velocity = mj_data.actuator_velocity[act_i]
        lengthrange = model.actuator_lengthrange[act_i]
        acc0 = model.actuator_acc0[act_i]
        prmb = model.actuator_biasprm[act_i, :9]
        prmg = model.actuator_gainprm[act_i, :9]
        
        # Calculate gain and bias
        bias = mujoco.mju_muscleBias(length, lengthrange, acc0, prmb)
        gain = min(-1.0, mujoco.mju_muscleGain(length, velocity, lengthrange, acc0, prmg))
        
        # Calculate muscle force
        active_forces[act_i] = gain * activation
        passive_forces[act_i] = bias
        
    # Get moment arm
    AM = np.zeros((n_actuators, n_dof))
    mujoco.mju_sparse2dense(AM,
                            mj_data.actuator_moment.reshape(-1),
                            mj_data.moment_rownnz,
                            mj_data.moment_rowadr,
                            mj_data.moment_colind.reshape(-1))
    
    active_torque_contribution = AM * active_forces[:, np.newaxis]
    passive_torque_contribution = AM * passive_forces[:, np.newaxis]
    total_torque_contribution = active_torque_contribution + passive_torque_contribution
    torque_contributions = np.concatenate([active_torque_contribution, passive_torque_contribution, total_torque_contribution],
                                          axis = 0)
    torque_contribution_df = pd.DataFrame(torque_contributions, columns = dof_names)
    torque_contribution_df.insert(0, "types", np.repeat(["Active", "Passive", "Total"], n_actuators))
    torque_contribution_df.insert(0, "muscle", np.tile(actuator_names, 3))
    return torque_contribution_df

def calc_ID_result(model: mujoco.MjModel, data: mujoco.MjData, disable_constraint = True) -> pd.DataFrame:
    """
    Calculate inverse dynamics

    :param model: mujoco model
    :param data: mujoco data
    
    :return: Dataframe (index: kinds of torques, columns: dof_names )
    """
    n_dof = model.nv

    # DOF names
    dof_names = get_dof_names(model)

    # Inverse dynamics
    mujoco.mj_inverse(model, data)

    if disable_constraint:
        old_disableflags = model.opt.disableflags
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
        mujoco.mj_inverse(model, data)
        model.opt.disableflags = old_disableflags
    else:
        mujoco.mj_inverse(model, data)
    
    # Get torque
    torque_matrix = np.row_stack([
        data.qfrc_inverse,
        data.qfrc_smooth,
        data.qfrc_passive,
        data.qfrc_constraint
    ])

    # Result
    columns_names = ["Net_Inverse", "Inertia_Smooth", "Passive_Elastic", "Constraint_Force"]
    torque_df = pd.DataFrame(torque_matrix, index=columns_names, columns=dof_names)
    
    return torque_df
    
# Constraint
def enforce_equality_constraints(model: mujoco.MjModel,
                                 data: mujoco.MjData):
    """
    Enforce equality constraint existing in mujoco model
    
    :param model: mujoco model
    :param data: data
    """
    for i in range(model.neq):
        if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT and data.eq_active[i]:
            x_jnt = model.eq_obj2id[i]
            y_jnt = model.eq_obj1id[i]
            
            x_adr = model.jnt_qposadr[x_jnt]
            y_adr = model.jnt_qposadr[y_jnt]
            
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

def get_equality_active(model):
    if hasattr(model, "eq_active"):
        return model.eq_active
    elif hasattr(model, "eq_active0"):
        return model.eq_active0
        
# Simulation
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
    # Model configl
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

# Mucle
def get_muscle_forceLengths(mjc_model: mujoco.MjModel,
                            muscle_name: str,
                            related_joint_names: list[str],
                            joint_configs: pd.DataFrame,
                            activations: list[float],
                            timestep = 0.005) -> pd.DataFrame:
    """
    Get muscle force and lengths when joint configuration is given

    :param mjc_model: mujoco model
    :param muscle_name: muscle name
    :param related_joint_names: related joint names on muscle
    :param joint_configs: joint angle configurations
    :param activations: value to activate muscle
    :param timestep: time step for forward dynamics
    
    :return: musculotendon forces given configuration, musculotendon lengths given configuration 
    """
    # Constants
    n_activations, n_configs = len(activations), len(joint_configs)
    related_joint_configs = joint_configs[related_joint_names]
    
    # Initialize model
    mjc_data = mujoco.MjData(mjc_model)
    mjc_model.opt.timestep = timestep
    joints_idx = [mujoco.mj_name2id(mjc_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) for joint_name in related_joint_names]
    muscle_id = mujoco.mj_name2id(mjc_model, mujoco.mjtObj.mjOBJ_ACTUATOR, muscle_name)

    locked_joint_data = get_locked_joint_angle(mjc_model)

    # Calculate muscle forces & lengths
    mtu_forces = np.zeros((n_activations, n_configs))
    mtu_lengths = np.zeros((n_activations, n_configs))
    for act_i, act in enumerate(activations):
        mjc_data.ctrl[muscle_id] = act # set control signal
        mujoco.mj_step(mjc_model, mjc_data)

        for config_i in range(n_configs):
            # Set joint configuration
            joint_config = related_joint_configs.iloc[config_i]
            mjc_data.qpos[:] = 0
            mjc_data.qvel[:] = 0
            mjc_data.qpos[joints_idx] = joint_config.values

            dependent_joint_data = calc_dependent_joint_angle(mjc_model, related_joint_names, joint_config)
            mjc_data.qpos[dependent_joint_data["dep_joint_i"]] = dependent_joint_data["dep_joint_angle"]
            mjc_data.qpos[locked_joint_data["joint_i"]] = locked_joint_data["angle"]

            # Forward dynamics
            mujoco.mj_step(mjc_model, mjc_data)

            # Stack result
            mtu_forces[act_i, config_i] = mjc_data.actuator_force[muscle_id].copy()
            mtu_lengths[act_i, config_i] = mjc_data.actuator_length[muscle_id].copy()
    result = np.concatenate([mtu_forces, mtu_lengths], axis = 0).T
    result = pd.DataFrame(result, columns = ["actuator_force", "mtu_length"])
    return result

def get_muscle_forces(mjc_model: mujoco.MjModel,
                      muscle_name: str,
                      related_joint_names: list[str],
                      joint_configs: pd.DataFrame,
                      activations: list[float],
                      timestep = 0.005) -> pd.DataFrame:
    """
    Get muscle force and lengths when joint configuration is given

    :param mjc_model: mujoco model
    :param muscle_name: muscle name
    :param related_joint_names: related joint names on muscle
    :param joint_configs: joint angle configurations
    :param activations: value to activate muscle
    :param timestep: time step for forward dynamics
    
    :return: musculotendon forces given configuration, musculotendon lengths given configuration 
    """
    # Constants
    n_act, n_configs = len(activations), len(joint_configs)
    related_joint_configs = joint_configs[related_joint_names]
    
    # Initialize model
    mjc_data = mujoco.MjData(mjc_model)
    mjc_model.opt.timestep = timestep
    joints_idx = [mujoco.mj_name2id(mjc_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) for joint_name in related_joint_names]
    muscle_id = mujoco.mj_name2id(mjc_model, mujoco.mjtObj.mjOBJ_ACTUATOR, muscle_name)

    locked_joint_data = get_locked_joint_angle(mjc_model)

    # Calculate muscle forces & lengths
    mtu_act = np.zeros(n_act * n_configs)
    mtu_len = np.zeros(n_act * n_configs)
    mtu_active_force = np.zeros(n_act * n_configs)
    mtu_passive_force = np.zeros(n_act * n_configs)
    mtu_total_force = np.zeros(n_act * n_configs)

    for config_i in range(n_configs):
        # Set joint configuration
        joint_config = related_joint_configs.iloc[config_i]
        mjc_data.qpos[:] = 0
        mjc_data.qvel[:] = 0
        mjc_data.qpos[joints_idx] = joint_config.values

        dependent_joint_data = calc_dependent_joint_angle(mjc_model, related_joint_names, joint_config)
        if len(dependent_joint_data) > 0:
            mjc_data.qpos[dependent_joint_data["dep_joint_i"]] = dependent_joint_data["dep_joint_angle"]
            mjc_data.qpos[locked_joint_data["joint_i"]] = locked_joint_data["angle"]
    
        for act_i, act in enumerate(activations):
            mjc_data.ctrl[muscle_id] = act # set control signal
            mujoco.mj_forward(mjc_model, mjc_data)

            # Stack result
            data_i = config_i * n_act + act_i
            mtu_act[data_i] = act
            mtu_len[data_i] = mjc_data.actuator_length[muscle_id].copy()
            
            force_info = calc_muscle_force(model = mjc_model,
                                           data = mjc_data,
                                           muscle_name = muscle_name,
                                           activation = act)
            
            mtu_active_force[data_i] = force_info["active_force"]
            mtu_passive_force[data_i] = force_info["passive_force"]
            mtu_total_force[data_i] = force_info["total_force"]
            
    result = pd.DataFrame({
        "activation": mtu_act,
        "mtu_len": mtu_len,
        "active_force": mtu_active_force,
        "passive_force": mtu_passive_force,
        "total_force": mtu_total_force,
    })
    return result

def calc_muscle_force(model: mujoco.MjModel,
                      data: mujoco.MjData,
                      muscle_name: str,
                      activation: float) -> dict:
    """
    Calculate muscle force based on mujoco muscle parameter

    :param model: mujoco model
    :param data: model state
    :param muscle_name: actuator name
    :param activation: activation value

    return dictionary
        - k: active_force
        - k: passive_force
        - k: total_force
    """
    mujoco.mj_forward(model, data)
    
    # Actuator info
    actuator_names = [model.actuator(i).name for i in range(model.nu)]
    actuator_i = actuator_names.index(muscle_name)

    # Muscle info
    length = data.actuator_length[actuator_i]
    velocity = data.actuator_velocity[actuator_i]
    lengthrange = model.actuator_lengthrange[actuator_i]
    acc0 = model.actuator_acc0[actuator_i]
    prmb = model.actuator_biasprm[actuator_i, :9]
    prmg = model.actuator_gainprm[actuator_i, :9]

    # Get gain and bias
    gain = mujoco.mju_muscleGain(length, velocity, lengthrange, acc0, prmg)
    bias = mujoco.mju_muscleBias(length, lengthrange, acc0, prmb)
    
    return {
        "active_force" : gain * activation,
        "passive_force" : bias,
        "total_force" : gain * activation + bias,
    }

# Verification
def calc_actuator_force_manually(model: mujoco.MjModel,
                                 mj_data: mujoco.MjData) -> np.ndarray:
    """
    Calculate actuator force manually

    :param model: mujoco model
    :param mj_data: model state

    :return: actuator force generated on each joint
    """
    moment_arm_mat = get_moment_arm_matrix(model, mj_data)
    manual_qfrc = np.dot(mj_data.actuator_force * model.actuator_gear[:, 0], moment_arm_mat)
    return manual_qfrc

def calc_qfrc_inverse_manually(M,
                               qacc,
                               qfrc_bias,
                               qfrc_passive,
                               qfrc_constraint):
    """
    Calculate joint torque from basis torque and inertia information

    - (M * acceleration) + bias - passive - constraint
    
    :param M: inertia matrix
    :param qacc: acceleration acting on the joint
    :param qfrc_bias:
    :param qfrc_passive: torque from passive muscle element
    :param qfrc_constraint: torque from constraint
    """
    qfrc_inverse_manual = (M @ qacc) + qfrc_bias - qfrc_passive - qfrc_constraint
    return qfrc_inverse_manual

def calculate_muscle_force_manually(model,
                                    mj_data,
                                    activation = 1.0) -> pd.DataFrame:
    """
    Calculate muscle force

    - active force: gain * muscle activation
    - passive force: bias
    
    :param model: mujoco model
    :param mj_data: mujoco data
    :param activation: muscle activation for simulation

    return muscle forces (columns: muscle, index: Active, Passive)
    """
    # Constants
    n_actuators = model.nu
    actuator_names = np.array([model.actuator(i).name for i in range(model.nu)])
    
    # Update geometry information
    mujoco.mj_step1(model, mj_data)
    
    # Calculate active forces
    active_forces = np.zeros(n_actuators)
    passive_forces = np.zeros(n_actuators)
    for act_i in range(n_actuators):
        # Extract target muscle params
        length = mj_data.actuator_length[act_i]
        velocity = mj_data.actuator_velocity[act_i]
        lengthrange = model.actuator_lengthrange[act_i]
        acc0 = model.actuator_acc0[act_i]
        prmb = model.actuator_biasprm[act_i, :9]
        prmg = model.actuator_gainprm[act_i, :9]
        
        # Calculate gain and bias
        bias = mujoco.mju_muscleBias(length, lengthrange, acc0, prmb)
        gain = mujoco.mju_muscleGain(length, velocity, lengthrange, acc0, prmg)
        
        # Calculate muscle force
        active_forces[act_i] = gain * activation
        passive_forces[act_i] = bias
    muscle_forces = np.vstack([active_forces, passive_forces])
    
    muscle_forces_df = pd.DataFrame(muscle_forces, columns = actuator_names)
    muscle_forces_df.index = ["Active", "Passive"]
    return muscle_forces_df

# Viewer
def display_qpos_viewer(
    model_path,
    qposes,
    marker_group = 4,
    marker_default_class="marker",
    height=400,
    width=600,
    framewidth=0.1,
    framelength=1.0,
    init_lookat=(0, 0, 0),
    init_camera=(2.5, -90, -90),
):
    # Model
    model = mujoco.MjModel.from_xml_path(model_path)
    model.vis.scale.framewidth = framewidth
    model.vis.scale.framelength = framelength
    data = mujoco.MjData(model)

    # Marker group from XML
    tree = ET.parse(model_path)
    root = tree.getroot()
    try:
        marker_default = root.find(f".//default[@class='{marker_default_class}']/site")
        marker_group = int(marker_default.get("group"))
    except:
        marker_group = marker_group

    # Render
    renderer = mujoco.Renderer(model, height=height, width=width)

    # Camera
    camera = mujoco.MjvCamera()

    # Scene option
    scene_option = mujoco.MjvOption()
    scene_option.frame = mujoco.mjtFrame.mjFRAME_WORLD
    scene_option.sitegroup[:] = 0
    scene_option.sitegroup[marker_group] = 1

    output_area = widgets.Output()

    # Widgets
    img_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(qposes) - 1,
        step=1,
        description="img:",
    )

    x_slider = widgets.FloatSlider(
        value=init_lookat[0],
        min=-1.0,
        max=1.0,
        step=0.01,
        description="look X:",
    )
    y_slider = widgets.FloatSlider(
        value=init_lookat[1],
        min=-1.0,
        max=1.0,
        step=0.01,
        description="look Y:",
    )
    z_slider = widgets.FloatSlider(
        value=init_lookat[2],
        min=-1.0,
        max=1.0,
        step=0.01,
        description="look Z:",
    )

    c_dist = widgets.FloatSlider(
        value=init_camera[0],
        min=0.0,
        max=10.0,
        step=0.1,
        description="Cam Dist",
        continuous_update=False,
    )
    c_azim = widgets.FloatSlider(
        value=init_camera[1],
        min=-180,
        max=180,
        step=1,
        description="Cam Azim",
        continuous_update=False,
    )
    c_elev = widgets.FloatSlider(
        value=init_camera[2],
        min=-90,
        max=90,
        step=1,
        description="Cam Elev",
        continuous_update=False,
    )

    def update(i, x, y, z, cam_dist, cam_azim, cam_elev):
        data.qpos[:] = qposes[i]
        mujoco.mj_forward(model, data)

        camera.lookat[:] = [x, y, z]
        camera.distance = cam_dist
        camera.azimuth = cam_azim
        camera.elevation = cam_elev

        renderer.update_scene(
            data,
            scene_option=scene_option,
            camera=camera,
        )

        pixels = renderer.render()

        with output_area:
            clear_output(wait=True)
            mediapy.show_image(pixels)

    ui = widgets.VBox([
        img_slider,
        widgets.HBox([x_slider, y_slider, z_slider]),
        widgets.HBox([c_dist, c_azim, c_elev]),
    ])

    out = widgets.interactive_output(
        update,
        {
            "i": img_slider,
            "x": x_slider,
            "y": y_slider,
            "z": z_slider,
            "cam_dist": c_dist,
            "cam_azim": c_azim,
            "cam_elev": c_elev,
        },
    )

    display(widgets.VBox([ui, output_area, out]))

