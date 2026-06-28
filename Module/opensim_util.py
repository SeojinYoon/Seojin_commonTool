
# Common Libraries
import os, re, sys, opensim, itertools, xarray
import numpy as np
import pandas as pd
import opensim as osim
from pathlib import Path
import xml.dom.minidom as md
import xml.etree.ElementTree as ET

# Custom Libraries
from sj_datastructure import make_3d_dataset
from XML.xml_util import search_tags_in_xml
from sj_array import get_ACS_explicit_orientation

# TRC
def convert_to_trc_coordinate(estim_3D: xarray.Dataset, 
                              target_markers: list[str]) -> tuple:
    """
    Convert 3D coordinates from xarray.Dataset into TRC-compatible numpy arrays.

    :param estim_3D: Input Dataset with dimensions (Times, Labels, Coords).
    :param target_markers: List of marker names to include (e.g., ["S", "E", "W", "EF"]).
    
    :return: A tuple of (processed_3d_array, flattened_trc_array).
             - processed_3d_array: Shape (n_time, n_marker, 3)
             - flattened_trc_array: Shape (n_time, n_marker * 3)
    :rtype: tuple
    """
    
    # 1. Select and Copy to avoid mutating the original Dataset
    selected_ds = estim_3D.sel(Labels = target_markers).copy()
    
    # 3. Ensure fixed dimension order (Time, Marker, Coord)
    # This prevents errors if the input dataset has a different axis order
    ordered_da = selected_ds["3D"].transpose("Times", "Labels", "Coords")
    processed_data = ordered_da.to_numpy()
    
    # 4. Reshape for OpenSim TRC Format
    # OpenSim expects each row to be one time step: [X1, Y1, Z1, X2, Y2, Z2, ...]
    n_time = processed_data.shape[0]
    n_marker = processed_data.shape[1]
    n_coord = processed_data.shape[2]
    
    # Shape becomes (n_time, n_marker * 3)
    data_trc = processed_data.reshape(n_time, n_marker * n_coord)
    
    return processed_data, data_trc
    
def numpy_to_trc(data, labels, trc_file_path, time):
    """
    Convert numpy arrays into trc file (meters) and save it.
    """
    # 시간 및 주파수 계산
    freq = 1 / (time[1] - time[0])
    n_frames = data.shape[0]
    frames = np.arange(1, n_frames + 1) # 프레임 번호는 보통 1부터 시작
    
    n_markers = len(labels)
    ncoords = 3
    
    # 헤더 작성 (Units를 m으로 변경)
    header = '\n'.join([
        'PathFileType\t4\t(X/Y/Z)\t{}'.format(trc_file_path),
        'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames',
        '{:.1f}\t{:.1f}\t{}\t{}\tm\t{:.1f}\t1\t{}'.format(freq, freq, n_frames, n_markers, freq, n_frames),
        'Frame#\tTime\t' + '\t\t\t'.join(labels),
        '\t\t' + '\t'.join(['X{}\tY{}\tZ{}'.format(i, i, i) for i in range(1, n_markers + 1)])
    ]) + '\n'
    
    # 데이터 결합 (Frame#, Time, X1, Y1, Z1, ...)
    kine = np.c_[frames, time, data.reshape(n_frames, n_markers * ncoords)]
    
    with open(trc_file_path, 'w') as output:
        output.write(header)
        # 포맷팅: Frame#는 정수(%d), 나머지는 소수점 아래 6자리(%.6f)로 정밀도 확보
        fmt = ['%d', '%.4f'] + (n_markers * ncoords * ['%.6f'])
        np.savetxt(output, kine, delimiter='\t', fmt=fmt)

    print(f"TRC file is saved (in meters): {trc_file_path}")

def trc_to_dataset(trc_path: str,
                   marker_names: list[str],
                   orientation: list[str]):
    """
    Converts a TRC file into a structured 3D dataset.

    :param trc_path: Path to the .trc file containing motion capture data.
    :param marker_names: A list of marker labels to extract from the TRC data.
    :param orientation: orientation of the data ex) ["LR", "IS", "AP"]
    
    :return: A structured 3D dataset object (exp_marker_ds).
    """
    # Load raw TRC data and extract time sequence
    trc_data = load_trc_file(trc_path)
    times = trc_data["Time"].to_numpy()
    
    # Remove metadata columns to isolate positional data
    pos_only = trc_data.drop(["Frame#", "Time"], axis = 1, level = 0)

    # Marker position
    exp_marker_pos = []
    for marker_name in marker_names:
        exp_marker_pos.append(pos_only[marker_name].to_numpy())
    
    # Convert list to array and transpose to shape: (Time, Markers, Coordinates)
    exp_marker_pos = np.array(exp_marker_pos).transpose(1, 0, 2)
    
    # Create the final 3D dataset with coordinates and metadata
    exp_marker_ds = make_3d_dataset(exp_marker_pos, 
                                    "3D", 
                                    element_dataset_names = ["Times", "Labels", "Coords"], 
                                    dataset1_dim_names = times, 
                                    dataset2_dim_names = marker_names,
                                    dataset3_dim_names = orientation)
    return exp_marker_ds
    
# Marker
def get_marker_positions(model: osim.simulation.Model,
                         state: osim.simbody.State,
                         marker_names: list):
    marker_locs = []
    for marker_name in marker_names:
        marker = model.getMarkerSet().get(marker_name)
        position = marker.getLocationInGround(state)
        position = np.array([position[0], position[1], position[2]])
        marker_locs.append(position)
    
    model_marker_pos = np.array(marker_locs)
    return model_marker_pos

def model_marker_positions(model: osim.simulation.Model,
                           state: osim.simbody.State,
                           marker_names: list , 
                           orientation: list[str]):
    """
    Generates a 3D dataset by extracting marker positions from a specified model.

    :param model: opensim model
    :param state: state
    :param marker_names: A list of marker labels/names to extract coordinates for.
    :param orientation: orientation of the data ex) ["LR", "IS", "AP"]
    
    :return: A structured 3D dataset object containing times, labels, and coordinates.
    """
    
    model_marker_pos = get_marker_positions(model, state, marker_names)

    # Convert list to NumPy array and add a batch dimension (index 0)
    model_marker_pos = np.expand_dims(model_marker_pos, 0)
    
    # Construct the final 3D dataset with metadata
    model_marker_pos_ds = make_3d_dataset(model_marker_pos, 
                                          "3D", 
                                          element_dataset_names = ["Times", "Labels", "Coords"], 
                                          dataset1_dim_names = [0], 
                                          dataset2_dim_names = marker_names,
                                          dataset3_dim_names = get_ACS_explicit_orientation(orientation))
    return model_marker_pos_ds
    
# Joint
def get_independent_joints(osim_model_path):
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

def get_joint_ranges(model_path):
    text = Path(model_path).read_text(errors="replace")
    
    start = text.find("<JointSet")
    end = text.find("</JointSet>", start) + len("</JointSet>")
    
    jointset_xml = "<root>" + text[start:end] + "</root>"
    root = ET.fromstring(jointset_xml)
    
    data = []
    
    objects = root.find(".//objects")
    
    for obj in list(objects):
        obj_name = obj.attrib.get("name")
        obj_type = obj.tag
    
        for coord in obj.findall(".//Coordinate"):
            joint_name = coord.attrib.get("name")
            range_elem = coord.find("range")
    
            if range_elem is not None and range_elem.text:
                min_val, max_val = map(float, range_elem.text.split())
    
                data.append({
                    "obj_name": obj_name,
                    "obj_type": obj_type,
                    "joint_name": joint_name,
                    "min": min_val,
                    "max": max_val
                })
    
    df = pd.DataFrame(data)
    return df

def extract_default_angles(osim_path):
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
    
# Muscle
def get_muscle_only(xml, target_information = "max_isometric_force"):
    """
    Get muscle informations
    
    :param xml(xml.etree.ElementTree.Element): xml
    
    return (pd.DataFrame)
    """
    # Find all muscles
    muscles = xml.findall(".//Millard2012EquilibriumMuscle")
    
    # Muscle dataframe
    max_isometric_forces = []
    muscle_names = []
    for muscle in muscles:
        muscle_name = muscle.get("name")
        max_isometric_force = muscle.find(target_information).text

        muscle_names.append(muscle_name)
        max_isometric_forces.append(float(max_isometric_force))
    data = pd.DataFrame(max_isometric_forces)
    data.index = muscle_names
    data.columns = [target_information]
    
    # Ceiling up to second decimal place
    data = data.applymap(lambda x: np.ceil(x * 100) / 100)
    
    return data
    
def extract_muscle_names(osim_file_path, tag = "Millard2012EquilibriumMuscle"):
    tree = ET.parse(osim_file_path)
    root = tree.getroot()
    return [muscle.get("name") for muscle in root.iter(tag)]

def extract_muscle_params(model_path, target_tag_name = "Millard2012EquilibriumMuscle"):
    muscle_names = extract_muscle_names(model_path, target_tag_name)
    
    tree = ET.parse(model_path)
    root = tree.getroot()
    
    data_info = {}
    for muscle_name in muscle_names:
        muscle_tag = root.find(f".//{target_tag_name}[@name='{muscle_name}']")
        
        for child in muscle_tag:
            tag_name = child.tag
            if "Curve" not in tag_name:
                val = child.text.strip() if child.text is not None else ""
                try: val = float(val) if '.' in val else int(val)
                except ValueError: pass
                data_info[(muscle_name, "Basic", tag_name)] = val
            elif "Curve" in tag_name:
                curve_root = muscle_tag.find(tag_name)        
                for curve_child in curve_root:
                    val = curve_child.text.strip() if curve_child.text is not None else ""
                    try: val = float(val) if '.' in val else int(val)
                    except ValueError: pass
                        
                    data_info[(muscle_name, tag_name, curve_child.tag)] = val
    series_data = pd.Series(data_info)
    df = series_data.unstack(level=[0])
    return df

# Scaling
def make_scale_setup(setup_template_path,
                     setup_file_path, 
                     template_model_path, 
                     pose_trc_path, 
                     time_range,
                     model_output_path,
                     mass = None,
                     is_preserver_mass_distribution = False):
    """
    Make scale setup file
    
    :param setup_template_path(string): scaling setup scaling template path
    :param setup_file_path(string): xml file path to save scaling setup file
    :param template_model_path(string): template opensim musculoskeletal model path
    :param pose_trc_path(string): static pose path (.trc)
    :param dynamic_trial_time_range(tuple - int): time range of dynmaic scaling
    :param model_output_path(string): output model path
    """
    # Copy template
    os.system(f"cp {setup_template_path} {setup_file_path}")
    print(f"Copy {setup_template_path} -> {setup_file_path}")
    
    # Root
    file = ET.parse(setup_file_path)
    root = file.getroot()

    # Set model template
    model_file_element = search_tags_in_xml(root, ["ScaleTool", "GenericModelMaker", "model_file"])
    assert len(model_file_element) == 1, "Error - model_file"
    model_file_element[0].text = template_model_path
    
    # Additional markers on model
    additional_marker_element = search_tags_in_xml(root, ["ScaleTool", "GenericModelMaker", "marker_set_file"])
    assert len(additional_marker_element) == 1, "Error - marker_set_file"
    additional_marker_element[0].text = "Unassigned"
    
    # Scale factor
    scaling_markers_element = search_tags_in_xml(root, ["ScaleTool", "ModelScaler", "marker_file"])
    assert len(scaling_markers_element) == 1, "Error - marker_file"
    scaling_markers_element[0].text = pose_trc_path
    
    scaling_markers_timeRange = search_tags_in_xml(root, ["ScaleTool", "ModelScaler", "time_range"])
    assert len(scaling_markers_timeRange) == 1, "Error - marker_file time range"
    scaling_markers_timeRange[0].text = f"{time_range[0]} {time_range[1]}"

    # Mass
    mass = -1 if mass is None else mass
    mass_tag = search_tags_in_xml(root, ["ScaleTool", "mass"])
    assert len(mass_tag) == 1, "Error - mass"
    mass_tag[0].text = f"{mass}"

    is_preserver_mass_distribution = "true" if is_preserver_mass_distribution else "false"
    scaling_preserve_mass_distribution = search_tags_in_xml(root, ["ScaleTool", "ModelScaler", "preserve_mass_distribution"])
    assert len(scaling_preserve_mass_distribution) == 1, "Error - preserver mass distribution"
    scaling_preserve_mass_distribution[0].text = is_preserver_mass_distribution
    
    # Marker place
    marker_pos_element = search_tags_in_xml(root, ["ScaleTool", "MarkerPlacer", "marker_file"])
    assert len(marker_pos_element) == 1, "Error - marker_file"
    marker_pos_element[0].text = pose_trc_path

    marker_pos_time_range = search_tags_in_xml(root, ["ScaleTool", "MarkerPlacer", "time_range"])
    assert len(marker_pos_time_range) == 1, "Error - marker_file time range"
    marker_pos_time_range[0].text = f"{time_range[0]} {time_range[1]}"
    
    # Output - scaled model
    output_model_scaler_element = search_tags_in_xml(root, ["ScaleTool", "ModelScaler", "output_model_file"])
    assert len(output_model_scaler_element) == 1, "Error - output_model_file"
    output_model_scaler_element[0].text = model_output_path
    
    output_marker_placer_element = search_tags_in_xml(root, ["ScaleTool", "MarkerPlacer", "output_model_file"])
    assert len(output_marker_placer_element) == 1, "Error - output_model_file"
    output_marker_placer_element[0].text = model_output_path

    # Write model
    with open(setup_file_path, "wb") as f:
        file.write(f, encoding="utf-8", xml_declaration=True)
        
    print(f"Scaling setup file is created completely: {setup_file_path}")
    
    return setup_file_path
    
# TRC
def load_trc_file(file_path, indicator='Frame#'):
    """
    Load trc file
    
    :param file_path(string): path for trc
    
    return (pd.DataFrame)
    """
    # Find the start of data and header line
    skip_rows1, column_names1 = find_data_start(file_path, indicator=indicator)
    skip_rows2, column_names2 = find_data_start(file_path, indicator="X1")
    
    for i, item in enumerate(column_names1):
        if item == "":
            column_names1[i] = column_names1[i-1]
    column_names1.append(column_names1[-1])
    column_names1.append(column_names1[-1])
    
    column_names2.insert(0, "")
    column_names2.insert(0, "")
            
    columns = pd.MultiIndex.from_arrays([column_names1, column_names2])
    # Load the .trc file with detected skiprows and use the column names
    data = pd.read_csv(file_path, delimiter='\t', skiprows = skip_rows2)
    data.columns = columns

    return data

def find_data_start(file_path, indicator='Frame#'):
    """
    Scan the .trc file to find where the data section starts.

    :param file_path(string): The path to the .trc file.
    :param indicator(string): A string that indicates the start of the data section. 
                 This is usually the first column name of the actual data.

    returns:
        - The line number where the data section starts, which can be used as skiprows in pd.read_csv.
        - The line containing column names, to be used as header.
    """
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            # Check if the indicator is in the current line
            if indicator in line:
                # Return the line number (i) and the previous line for column names
                # Adjust the return value based on whether your file contains an extra line
                # between the column names and the first row of data
                return i, line.strip().split('\t')
    return 0, None  # Return 0, None if the indicator is not found
    
# Inverse kinematics
def make_inverse_kinematics_setup(setup_template_path,
                                  setup_file_path, 
                                  model_path, 
                                  trc_path,
                                  ik_output_dir_path):
    """
    Make inverse kinematics setup file (xml file)
    
    :param setup_template_path(string): inverse kinematics setup inverse kinematics template path
    :param setup_file_path(string): xml file path to save inverse kinematics setup file
    :param model_path(string): opensim skeletal-musculo model path
    :param trc_path(string): trc file path
    :param ik_output_dir_path(string): output path of inverse kinematics
    """
    # Make directory
    os.makedirs(ik_output_dir_path, exist_ok = True)

    # Copy basic setup file
    destination = setup_file_path
    os.system(f"cp {setup_template_path} {destination}")
    print(f"Copy {setup_template_path} -> {destination}")
    
    # Output file path
    output_file_name = os.path.join(ik_output_dir_path, f"{Path(trc_path).stem}_IK.mot")
    
    # Change setup
    file = md.parse(setup_file_path)
    file.getElementsByTagName("model_file")[0].childNodes[0].nodeValue = model_path
    file.getElementsByTagName("results_directory")[0].childNodes[0].nodeValue = ik_output_dir_path
    file.getElementsByTagName("marker_file")[0].childNodes[0].nodeValue = trc_path
    file.getElementsByTagName("output_motion_file")[0].childNodes[0].nodeValue = output_file_name
    
    # Writing the changes in file
    with open(setup_file_path, "w") as fs:
        fs.write(file.toxml())
        fs.close()
    
    print(f"Inverse kinematics setup file is created completely: {setup_file_path}")
    
    return destination
    
def load_ik_log(file_path):
    pattern = re.compile(r"Frame (\d+) \(t = ([\d\.]+)\):.*RMS = ([\d\.]+), max = ([\d\.]+) \((.*?)\)")
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                frame = int(match.group(1))
                time = float(match.group(2))
                rms = float(match.group(3))
                max_err = float(match.group(4))
                marker = match.group(5)
                
                data.append([frame, time, rms, max_err, marker])
    df = pd.DataFrame(data, columns=["frame", "time", "RMS", "max_error", "max_marker"])
    return df

# Read opensim datas
def read_mot_file(file_path):
    """
    Read motion file
    
    :parma file_path(string): file path of .mot
    
    return header, dataframe
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    header_end_idx = lines.index('endheader\n')
    column_names = lines[header_end_idx + 1].strip().split('\t')
    
    df_mot = pd.read_csv(file_path, sep='\t', skiprows = header_end_idx + 2, names=column_names)
    
    header = " ".join(lines[:header_end_idx])
    return header, df_mot

def readStoFile(file_path):
    """
    read .sto file
    
    :params file_path(string): .sto file path
    
    return (list, pd.DataFrame) 
    """

    if not os.path.exists(file_path):
        print('file do not exists')

    file_id = open(file_path, 'r')

    # read header
    next_line = file_id.readline()
    header = [next_line]
    nc = 0
    nr = 0
    while not 'endheader' in next_line:
        if 'datacolumns' in next_line:
            nc = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'datarows' in next_line:
            nr = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'nColumns' in next_line:
            nc = int(next_line[next_line.index('=') + 1:len(next_line)])
        elif 'nRows' in next_line:
            nr = int(next_line[next_line.index('=') + 1:len(next_line)])

        next_line = file_id.readline()
        header.append(next_line)

    # process column labels
    next_line = file_id.readline()
    if next_line.isspace() == True:
        next_line = file_id.readline()

    labels = next_line.split()

    # get data
    data = []
    for i in range(1, nr + 1):
        d = [float(x) for x in file_id.readline().split()]
        data.append(d)

    file_id.close()

    df = pd.DataFrame(data)
    df.columns = labels
    
    return header, df
    
# Others
def mm_to_m(vector):
    """
    Convert mm to m
    
    :param vector(np.array): 
    
    return vector which has m unit
    """
    return vector / 1000

def m_to_mm(vector):
    """
    Convert m to mm
    
    :param vector(np.array): 
    
    return vector which has m unit
    """
    return vector * 1000

def convert_mot2sto(mot_file_path, save_file_path):
    """
    Convert mot file into sto file
    
    :param mot_file_path(string): file path of .mot
    :param save_file_path(string): file path of .sto
    
    return dataframe
    """
    header, df_mot = read_mot_file(mot_file_path)
    columns = df_mot.columns.copy()
    columns = list(columns)
    
    for column in ["time", "t_x", "t_y", "t_z"]:
        columns.remove(column)
    
    # Convert degree into Radian
    df_mot[columns] = df_mot[columns].apply(np.radians)
    
    # Prepare the header for the .sto file
    sto_header = [
        "nameOfFile\n",
        "version=1\n",
        f"nRows={df_mot.shape[0]}\n",
        f"nColumns={df_mot.shape[1]}\n",
        "inDegrees=no\n",  # Indicate that the file is now in radians
        "endheader\n"
    ]
    
    # Write the header and modified data to the .sto file
    with open(save_file_path, 'w') as sto_file:
        sto_file.writelines(sto_header)  # Write the header
        df_mot.to_csv(sto_file, sep='\t', index=False)  # Write data without the pandas header
        print("saved: ", save_file_path)
        
    return df_mot 

# CMC
def make_cmc_setup(setup_file_path, 
                   model_file_path,
                   cmc_initial_time,
                   cmc_final_time, 
                   analysis_start_time,
                   analysis_end_time,
                   kinematics_data_path,
                   taskSet_path,
                   constraint_path,
                   save_dir_path,
                   prefix,
                   lowpass_filtering = 6,
                   maximum_number_of_integrator_steps = 300000000,
                   maximum_integrator_step_size = 1,
                   minimum_integrator_step_size = 0.0001,
                   integrator_error_tolerance = 0.0005,
                   cmc_time_window = 0.01,
                   precision = 8):
    """
    Make CMC setup file
    
    :param setup_file_path(string): Path where the setup file will be saved.
    :param model_file_path(string): model file path
    :param cmc_initial_time(float): The starting time of the CMC simulation in seconds. (default: 0.03)
    :param cmc_final_time(float) The ending time of the CMC simulation in seconds.
    :param analysis_start_time(float): The starting time for analysis within the CMC simulation.
    :param analysis_end_time(float): The ending time for analysis within the CMC simulation.
    :param kinematics_data_path(string): Path to the file containing the desired kinematics data (e.g., from IK results).
    :param taskSet_path(string): Path to the file containing the task set definitions for the CMC simulation.
    :param constraint_path(string): Path to the file containing any constraints for the CMC simulation.
    :param save_dir_path(string): Directory where the results of the CMC simulation will be saved.
    :param prefix(str): Prefix for save result of CMC
    :param lowpass_filtering(int): Cutoff frequency for lowpass filtering applied to kinematics data.
    :param maximum_number_of_integrator_steps(int): Maximum number of steps the integrator is allowed to take during the simulation.
    :param maximum_integrator_step_size(float): Maximum step size for the integrator during the simulation.
    :param minimum_integrator_step_size(float): Minimum step size for the integrator during the simulation.
    :param integrator_error_tolerance(float): Error tolerance for the integrator during the simulation.
    :param cmc_time_window(float): Time window over which the CMC tool solves for muscle activations.
    :param precision(int): Precision for the output data files.
    """
    # Load xml
    file = ET.parse(setup_file_path)
    root = file.getroot()

    # Model
    model_tag = search_tags_in_xml(root, ["CMCTool", "model_file"])[0]
    model_tag.text = model_file_path
    
    # Prefix
    cmc_tag = search_tags_in_xml(root, ["CMCTool"])[0]
    cmc_tag.set("name", prefix)
    
    # Integrator settings
    max_number_integrator_step_tag = search_tags_in_xml(root, ["CMCTool", "maximum_number_of_integrator_steps"])[0]
    max_number_integrator_step_tag.text = str(maximum_number_of_integrator_steps)

    max_integrator_step_size_tag = search_tags_in_xml(root, ["CMCTool", "maximum_integrator_step_size"])[0]
    max_integrator_step_size_tag.text = str(maximum_integrator_step_size)

    minimum_integrator_step_size_tag = search_tags_in_xml(root, ["CMCTool", "minimum_integrator_step_size"])[0]
    minimum_integrator_step_size_tag.text = str(minimum_integrator_step_size)

    integrator_error_tolerance_tag = search_tags_in_xml(root, ["CMCTool", "integrator_error_tolerance"])[0]
    integrator_error_tolerance_tag.text = str(integrator_error_tolerance)
    
    # Time
    cmc_initial_time_tag = search_tags_in_xml(root, ["CMCTool", "initial_time"])[0]
    cmc_initial_time_tag.text = str(cmc_initial_time)
    cmc_final_time_tag = search_tags_in_xml(root, ["CMCTool", "final_time"])[0]
    cmc_final_time_tag.text = str(cmc_final_time)

    analysisSet_kinematics_start_time_tag = search_tags_in_xml(root, ["CMCTool", "AnalysisSet", "objects", "Kinematics", "start_time"])[0]
    analysisSet_kinematics_start_time_tag.text = str(analysis_start_time)
    analysisSet_kinematics_end_time_tag = search_tags_in_xml(root, ["CMCTool", "AnalysisSet", "objects", "Kinematics", "end_time"])[0]
    analysisSet_kinematics_end_time_tag.text = str(analysis_end_time)

    analysisSet_actuation_start_time_tag = search_tags_in_xml(root, ["CMCTool", "AnalysisSet", "objects", "Actuation", "start_time"])[0]
    analysisSet_actuation_start_time_tag.text = str(analysis_start_time)
    analysisSet_actuation_end_time_tag = search_tags_in_xml(root, ["CMCTool", "AnalysisSet", "objects", "Actuation", "end_time"])[0]
    analysisSet_actuation_end_time_tag.text = str(analysis_end_time)

    cmc_time_window_tag = search_tags_in_xml(root, ["CMCTool", "cmc_time_window"])[0]
    cmc_time_window_tag.text = str(cmc_time_window)
    
    # Input
    desired_kinematics_file_tag = search_tags_in_xml(root, ["CMCTool", "desired_kinematics_file"])[0]
    desired_kinematics_file_tag.text = kinematics_data_path

    task_set_file_tag = search_tags_in_xml(root, ["CMCTool", "task_set_file"])[0]
    task_set_file_tag.text = taskSet_path

    constraint_file_tag = search_tags_in_xml(root, ["CMCTool", "constraints_file"])[0]
    constraint_file_tag.text = constraint_path

    lf_tag = search_tags_in_xml(root, ["CMCTool", "lowpass_cutoff_frequency"])[0]
    lf_tag.text = str(lowpass_filtering)
    
    # Save directory
    result_directory_tag = search_tags_in_xml(root, ["CMCTool", "results_directory"])[0]
    result_directory_tag.text = save_dir_path

    precision_tag = search_tags_in_xml(root, ["CMCTool", "output_precision"])[0]
    precision_tag.text = str(precision)
    
    # Write model
    with open(setup_file_path, "wb") as f:
        file.write(f, encoding="utf-8", xml_declaration=True)
    
    return setup_file_path

# Geometry
def UpdateGeometryPathToMesh(musculoskeletal_model_path, geometry_path):
    """
    Update model file to add geometry path to mesh
    
    :param musculoskeletal_model_path(string): model path
    :param geometry_path(string): Geometry path
    """
    # Get xml files
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(musculoskeletal_model_path, parser)
    root = tree.getroot()

    # Find all meshs
    mesh_files = root.findall(".//mesh_file")

    # Update mesh path
    for mesh_file in mesh_files:
        if mesh_file.text:
            mesh_file.text = os.path.join(geometry_path, mesh_file.text)

    # Write the modifications back to the file, preserving comments
    tree.write(musculoskeletal_model_path, xml_declaration=True, encoding='UTF-8')

# Static optimization
def make_staticOptimization_setup(setup_template_path,
                                  setup_file_path,
                                  model_path,
                                  so_output_dir_path,
                                  ik_path,
                                  time_range):
    """
    Make static optimization setup file (xml file)

    :param setup_template_path(string): static optimization setup template path
    :param setup_file_path(string): xml file path to save static optimization setup file
    :param model_path(string): opensim skeletal-musculo model path
    :param so_output_dir_path(string): directory path for saving static optimization results
    :param ik_path(string): the static optimization's target... inverse kinematic result path
    :param time_range(tuple): static optimization time range (start_time, end_time) ex) (0, 5)
    """
    model_name = Path(model_path).stem

    # Make directory
    os.makedirs(so_output_dir_path, exist_ok = True)

    # Time range
    start, stop = time_range

    # Copy basic setup file
    destination = setup_file_path
    os.system(f"cp {setup_template_path} {destination}")
    print(f"Copy {setup_template_path} -> {destination}")
    
    # Change setup
    tree = ET.parse(setup_file_path)
    root = tree.getroot()
    root[0].set('name', model_name)
    tree.write(setup_file_path)

    file = md.parse(setup_file_path)
    file.getElementsByTagName("model_file")[0].childNodes[0].nodeValue = model_path
    file.getElementsByTagName("results_directory")[0].childNodes[0].nodeValue = so_output_dir_path
    file.getElementsByTagName("output_precision")[0].childNodes[0].nodeValue = 8
    file.getElementsByTagName("initial_time")[0].childNodes[0].nodeValue = start
    file.getElementsByTagName("final_time")[0].childNodes[0].nodeValue = stop
    file.getElementsByTagName("solve_for_equilibrium_for_auxiliary_states")[0].childNodes[0].nodeValue = False
    file.getElementsByTagName("maximum_number_of_integrator_steps")[0].childNodes[0].nodeValue = 2000
    file.getElementsByTagName("maximum_integrator_step_size")[0].childNodes[0].nodeValue = 1
    file.getElementsByTagName("minimum_integrator_step_size")[0].childNodes[0].nodeValue = 1e-08
    file.getElementsByTagName("integrator_error_tolerance")[0].childNodes[0].nodeValue = 1.0e-05
    file.getElementsByTagName("integrator_error_tolerance")[0].childNodes[0].nodeValue = 1.0e-05
    file.getElementsByTagName("activation_exponent")[0].childNodes[0].nodeValue = 2
    file.getElementsByTagName("coordinates_file")[0].childNodes[0].nodeValue = ik_path
    # file.getElementsByTagName("states_file")[0].childNodes[0].nodeValue = path_to_ID + '.sto'

    # writing the changes in "file"
    with open(setup_file_path, "w") as fs:
        fs.write(file.toxml())
        fs.close()

    print(f"Static optimization setup file is created completely: {setup_file_path}")
    
    return destination

def get_muscle_lengths(osim_model: opensim.simulation.Model,
                       osim_muscle: opensim.simulation.Muscle,
                       joint_configs: pd.DataFrame,
                       joint_names: [str]) -> (list, pd.DataFrame):
    """
    Get selected length of muscle from joint configuration

    :param osim_model: opensim model
    :param osim_muscle: opensim muscle
    :param joint_configs: joint angle configurations
    :param joint_names: joint name list
    
    return muscle tendon lengths
    """
    # Initialize model and get constants
    currentState = osim_model.initSystem()
    joints = [osim_model.getCoordinateSet().get(j) for j in joint_names]
    
    # Calculate muscle tendon lengths based on the meshes of joint angles
    n_rows = len(joint_configs)
    mtu_len = np.zeros(n_rows, dtype = np.float64)
    joint_values_matrix = joint_configs.values
    for i in range(n_rows):
        angles = joint_values_matrix[i]
        for osim_joint, angle in zip(joints, angles):
            osim_joint.setValue(currentState, angle)
            
        mtu_len[i] = osim_muscle.getGeometryPath().getLength(currentState)
        
    return mtu_len
    
def get_muscle_length_dist(osim_model: opensim.simulation.Model,
                           osim_muscle: opensim.simulation.Muscle,
                           joint_names: [str],
                           optimal_fiber_len: float,
                           tendon_slack_len: float,
                           joint_angle_ranges: pd.DataFrame,
                           n_eval_angle: int = 7,
                           n_dist_sampling: int = 15) -> (list, pd.DataFrame):
    """
    Get selected length of muscle from joint configuration

    :param osim_model: opensim model
    :param osim_muscle: opensim muscle
    :param joint_names: joint name list
    :param joint_angle_ranges: range of motion on each joint (#column: #joint, rows: min, max)
    :param n_eval_angle: the number of evaluation, which is used splitting angle on each joint
    :param n_dist_sampling: the number of sampling to construct distribution of joint configuration 
    
    return muscle tendon lengths, joint configurations
    """
    # Initialize model and get constants
    currentState = osim_model.initSystem()
    joints_idx = [osim_model.getCoordinateSet().get(j) for j in joint_names]
    
    # Make joint configurations
    ang_mesh = [np.linspace(joint_angle_ranges.loc["min", joint_name],
                            joint_angle_ranges.loc["max", joint_name], n_eval_angle) for joint_name in joint_names]
    joint_angle_configs = pd.DataFrame(list(itertools.product(*ang_mesh)), columns = joint_names)

    # Get musculotendon lengths
    mtu_lengths = get_muscle_lengths(osim_model = osim_model,
                                     osim_muscle = osim_muscle,
                                     joint_configs = joint_angle_configs,
                                     joint_names = joint_names)

    # Filter physiological plausible muscle length only
    physio_min_len = tendon_slack_len + (optimal_fiber_len * 0.5)
    physio_max_len = tendon_slack_len + (optimal_fiber_len * 1.5)
    
    margin = optimal_fiber_len * 0.2
    valid_range = (physio_min_len - margin, physio_max_len + margin)
    valid_muscle_length_idx = np.where((mtu_lengths >= valid_range[0]) & (mtu_lengths <= valid_range[1]))[0]
    
    filtered_muscle_lengths = mtu_lengths[valid_muscle_length_idx]
    filtered_joint_angle_configs = joint_angle_configs.iloc[valid_muscle_length_idx].reset_index(drop = True)

    # Generate the even distributed muscle tendon lenghts based on the min 
        # and max of muscle lengths that calculated from the mesh joint angles
    mtu_lengths_dis = np.linspace(filtered_muscle_lengths.min(), filtered_muscle_lengths.max(), n_dist_sampling)
    selected_indices = []
    for target_len in mtu_lengths_dis:
        # find the closed muscle tendon lengths from the even distributed mtu lengths
        abs_diff = np.abs(filtered_muscle_lengths - target_len)
        closest_idx = int(np.where(abs_diff == abs_diff.min())[0][0])
        selected_indices.append(closest_idx)
    
    final_mtu_lengths = filtered_muscle_lengths[selected_indices]
    final_joint_configs_df = filtered_joint_angle_configs.iloc[selected_indices].reset_index().drop("index", axis = 1)
    return final_mtu_lengths, final_joint_configs_df
                                         
def get_muscle_forces(osim_model: opensim.simulation.Model,
                      osim_muscle: opensim.simulation.Muscle,
                      joint_names: list[str],
                      joint_configs: pd.DataFrame,
                      act_list: list[float]) -> pd.DataFrame:
    """
    Get muscle forces when a muscle actives 

    :param osim_model: opensim model
    :param osim_muscle: opensim muscle
    :param joint_names: joint name list
    :param joint_configs: joint angle configurations
    :param act_list: muscle activation value to be simulated

    :return: dataframe including columns: musculutendon length, total muscle force, active muscle force, passive muscle force
    """
    # Initialize model
    currentState = osim_model.initSystem()
    joints = [osim_model.getCoordinateSet().get(j) for j in joint_names]
    
    # Calculate equilibrated muscle force from the selected joint configurations
    n_config = len(joint_configs)
    n_act = len(act_list)

    mtu_act = np.zeros(n_act * n_config)
    mtu_ang = np.zeros(n_act * n_config)
    mtu_len = np.zeros(n_act * n_config)
    mtu_active_force = np.zeros(n_act * n_config)
    mtu_passive_force = np.zeros(n_act * n_config)
    mtu_total_force = np.zeros(n_act * n_config)
    for config_i, joint_angle_config in enumerate(joint_configs.values):
        for osim_joint, angle in zip(joints, joint_angle_config):
            osim_joint.setValue(currentState, angle)
    
        for act_i, activation in enumerate(act_list):
            osim_muscle.setActivation(currentState, activation)
            osim_model.equilibrateMuscles(currentState)

            data_i = config_i * n_act + act_i
            mtu_act[data_i] = activation
            mtu_ang[data_i] = osim_muscle.getPennationAngle(currentState)
            mtu_len[data_i] = osim_muscle.getGeometryPath().getLength(currentState)
            mtu_active_force[data_i] = osim_muscle.getActiveFiberForce(currentState)
            mtu_passive_force[data_i] = osim_muscle.getPassiveFiberForce(currentState)
            mtu_total_force[data_i] = osim_muscle.getFiberForceAlongTendon(currentState)
    
    result = pd.DataFrame({
        "activation": mtu_act,
        "pennation_ang" : mtu_ang,
        "mtu_len": mtu_len,
        "active_force": mtu_active_force,
        "passive_force": mtu_passive_force,
        "total_force": mtu_total_force,
    })
    
    return result
    
if __name__=="__main__":
    readStoFile("")
    read_mot_file("")

    s = convert_mot2sto(os.path.join(dir_path, "cut1.mot"), "/home/seojin/test/test.sto")
    