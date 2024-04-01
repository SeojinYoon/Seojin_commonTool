
# Common Libraries
from pathlib import Path
import xml.etree.ElementTree as ET
import xml.dom.minidom as md
import os
import sys

# Custom Libraries
from XML.xml_util import search_tags_in_xml

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
    
def run_scaling(setup_file_path,
                docker_name = "seojin_opensim"):
    """
    Run inverse kinematics with an xml setup file
    
    :param setup_file_path(string): xml file path to save scaling setup file
    :param docker_name(string): the name of docker
    """
    from sj_docker import run_command_onDocker
    
    command = f"opensim-cmd run-tool {setup_file_path}"
    output = run_command_onDocker(command, docker_name = docker_name)
    
    print(f"Model is created completely")
    
    return output

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

def run_inverseKinematics(setup_template_path,
                          setup_file_path,
                          model_path,
                          trc_path,
                          ik_output_dir_path,
                          docker_name = "seojin_opensim"):
    """
    Run inverse kinematics with an xml setup file
    
    :param setup_template_path(string): inverse kinematics setup inverse kinematics template path
    :param setup_file_path(string): xml file path to save static optimization  setup file
    :param model_path(string): opensim skeletal-musculo model path
    :param trc_path(string): trc file path
    :param ik_output_dir_path(string): output path of inverse kinematics
    :param docker_name(string): the name of docker
    """
    from sj_docker import run_command_onDocker
    
    make_inverse_kinematics_setup(setup_template_path = setup_template_path,
                                  setup_file_path = setup_file_path, 
                                  model_path = model_path, 
                                  trc_path = trc_path,
                                  ik_output_dir_path = ik_output_dir_path)

    command = f"opensim-cmd run-tool {setup_file_path}"
    output = run_command_onDocker(command, docker_name = docker_name)
    
    return output

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

    print("Static optimization setup file is created completely: {setup_file_path}")

def run_staticOptimization(setup_template_path,
                           setup_file_path,
                           model_path,
                           so_output_dir_path,
                           time_range,
                           ik_path,
                           docker_name = "seojin_opensim"):
    """
    Run static optimization with an xml setup file
    
    :param setup_template_path(string): static optimization setup template path
    :param setup_file_path(string): xml file path to save static optimization setup file
    :param model_path(string): opensim skeletal-musculo model path
    :param so_output_dir_path(string): directory path for saving static optimization results
    :param time_range(tuple): static optimization time range (start_time, end_time) ex) (0, 5)
    :param ik_path(string): the static optimization's target... inverse kinematic result path
    :param docker_name(string): the name of docker
    """
    
    from sj_docker import run_command_onDocker
    
    make_staticOptimization_setup(setup_template_path = setup_template_path,
                                  setup_file_path = setup_file_path, 
                                  model_path = model_path, 
                                  so_output_dir_path = so_output_dir_path, 
                                  ik_path = ik_path,
                                  time_range = time_range)
    
    command = f"opensim-cmd run-tool {setup_file_path}"
    output = run_command_onDocker(command = command, docker_name = docker_name)
    
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
    muscles = root.findall(".//Millard2012EquilibriumMuscle")
    
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

def read_mot_file(file_path):
    """
    Read motion file
    
    :parma file_path(string): file path of .mot
    
    return header, dataframe
    """
    with open(mot_file_path, 'r') as file:
        lines = file.readlines()
    header_end_idx = lines.index('endheader\n')
    column_names = lines[header_end_idx + 1].strip().split('\t')
    
    df_mot = pd.read_csv(mot_file_path, sep='\t', skiprows = header_end_idx + 2, names=column_names)
    
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

def get_marker_position(model_file, marker_name):
    """
    Get the position of a specific marker in an OpenSim model.
    
    :param model_file(string): Path to the OpenSim model file (.osim)
    :param marker_name(string): The name of the marker whose position you want to retrieve
    
    return: The position of the marker as an OpenSim Vec3 object
    """
    import opensim as osim
    
    # Load the model
    model = osim.Model(model_file)
    model.initSystem()

    # Access the marker set and the specific marker by name
    marker = model.getMarkerSet().get(marker_name)

    # Get the marker's position in the global frame
    # Note: initSystem() or realizePosition() might be needed to ensure
    # the model is in a consistent state for position retrieval
    position = marker.get_location_in_ground(model.initSystem())

    return position

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