
# Common Libraries
from pathlib import Path
import xml.etree.ElementTree as ET
import xml.dom.minidom as md
import os
import sys
import numpy as np
import pandas as pd

# Custom Libraries
from XML.xml_util import search_tags_in_xml
from sj_docker import run_command_onDocker

x_index = 0
y_index = 1
z_index = 2

# Scale model
def make_scale_setup(setup_template_path,
                     setup_file_path, 
                     template_model_path, 
                     static_pose_trc_path, 
                     dynamic_trial_trc_path,
                     dynamic_trial_time_range,
                     model_output_path):
    """
    Make scale setup file
    
    :param setup_template_path(string): scaling setup scaling template path
    :param setup_file_path(string): xml file path to save scaling setup file
    :param template_model_path(string): template opensim musculoskeletal model path
    :param static_pose_trc_path(string): static pose path (.trc)
    :param dynamic_trial_trc_path(string): dynamic trial trc path
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
    
    # Scale factor - dynamic scaling 
    dynamic_scaling_markers_element = search_tags_in_xml(root, ["ScaleTool", "ModelScaler", "marker_file"])
    assert len(dynamic_scaling_markers_element) == 1, "Error - marker_file"
    dynamic_scaling_markers_element[0].text = dynamic_trial_trc_path
    
    dynamic_scaling_markers_timeRange = search_tags_in_xml(root, ["ScaleTool", "ModelScaler", "time_range"])
    assert len(dynamic_scaling_markers_timeRange) == 1, "Error - marker_file time range"
    dynamic_scaling_markers_timeRange[0].text = f"{dynamic_trial_time_range[0]} {dynamic_trial_time_range[1]}"
    
    # Marker place
    marker_pos_element = search_tags_in_xml(root, ["ScaleTool", "MarkerPlacer", "marker_file"])
    assert len(marker_pos_element) == 1, "Error - marker_file"
    marker_pos_element[0].text = static_pose_trc_path
    
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

# Run opensim-cmd
def run_opensim_cmd(setup_file_path,
                    docker_name = "seojin_opensim2"):
    """
    Run opensim-cmd
    
    :param setup_file_path(string): xml file path to save scaling setup file
    :param docker_name(string): the name of docker
    """
    command = f"opensim-cmd run-tool {setup_file_path}"
    output = run_command_onDocker(command, docker_name = docker_name)
    
    return output

# Update
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

# Information
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

# Marker positions
def get_marker_position(model_path, marker_name):
    """
    Get the position of a specific marker in an OpenSim model.
    
    :param model_file(string): Path to the OpenSim model file (.osim)
    :param marker_name(string): The name of the marker whose position you want to retrieve
    
    return: The position of the marker as an OpenSim Vec3 object
    """
    import opensim as osim
    
    # Load the model
    model = osim.Model(model_path)
    state = model.initSystem()

    # Access the marker set and the specific marker by name
    marker = model.getMarkerSet().get(marker_name)

    # Get the marker's position in the global frame
    position = marker.getLocationInGround(state)

    return position

def get_body_positions(model_path):
    """
    Get body positions from model
    
    :param model_path(string): path of model
    
    return (dictionary)
        -k bodyname
            -k locationInGround: location relative to ground frame
    """
    import opensim as osim
    
    model = osim.Model(model_path)
    state = model.initSystem() # initial state of model
    
    # Get body positions
    body_position_info = {}
    for body in bodySet:
        body_location = body.getPositionInGround(state)
        body_location = np.array([body_location[x_index], 
                                  body_location[y_index], 
                                  body_location[z_index]])
        body_position_info[body.getName()] = {
            "locationInGround" : body_location
        }
    return body_position_info
        
def get_marker_positions(model_path):
    """
    Get marker positions from model
    
    :param model_path(string): path of model
    
    return (dictionary)
        -k marker_name
            -k parent: parent frame name
            -k locationInGround: location relative to ground frame
            -k locationInParent: location relative to parent frame
    """
    import opensim as osim
    
    model = osim.Model(model_path)
    state = model.initSystem() # initial state of model
    
    marker_position_info = {}
    for marker in markerSet:
        marker_name = marker.getName()
        parent_frame = marker.getParentFrame()
        parent_frame_name = parent_frame.getName()
        parent_frame.getPositionInGround(state)

        # Get marker location - from parent frame
        marker_location_in_parent = marker.findLocationInFrame(state, parent_frame)
        marker_location_in_parent = np.array([marker_location_in_parent[x_index], 
                                              marker_location_in_parent[y_index],
                                              marker_location_in_parent[z_index]])

        # Get marker location - from ground frame
        marker_locationInGround = marker.getLocationInGround(state)
        marker_locationInGround = np.array([marker_locationInGround[x_index], 
                                            marker_locationInGround[y_index], 
                                            marker_locationInGround[z_index]])

        marker_position_info[marker_name] = {
            "parentFrame_name" : parent_frame_name,
            "locationInGround" : marker_locationInGround,
            "locationInParent" : marker_location_in_parent,
        }
        
    return marker_position_info

# Others
def mm_to_m(vector):
    """
    Convert mm to m
    
    :param vector(np.array): 
    
    return vector which has m unit
    """
    return vector * 1000

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

if __name__=="__main__":
    readStoFile("")
    read_mot_file("")

    s = convert_mot2sto(os.path.join(dir_path, "cut1.mot"), "/home/seojin/test/test.sto")
    
    output = run_staticOptimization(template_path = staticOptimization_template_path,
                                    model_path = model_path,
                                    setup_file_path = "/mnt/sdb2/DeepDraw/Projects/20220801_DP02_mri/Opensim/test/setup/test.xml",
                                    ik_path = ik_path,
                                    so_output_dir_path = "/mnt/sdb2/DeepDraw/Projects/20220801_DP02_mri/Opensim/test/so_new",
                                    time_range = (0,5))
    
    load_trc_file("/mnt/sdb2/DeepDraw/Projects/20220801_DP02_mri/Opensim/TRC_files/resnet_50/trial/trial2.trc")