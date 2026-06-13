
# Common Libraries
import os, re, sys
import numpy as np
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET

# Custom Libraries
from sj_docker import run_command_onDocker
from XML.xml_util import search_tags_in_xml

x_index = 0
y_index = 1
z_index = 2

# Run opensim-cmd
def run_opensim_cmd(setup_file_path,
                    container_ID = "seojin_opensim2"):
    """
    Run opensim-cmd
    
    :param setup_file_path(string): xml file path to save scaling setup file
    :param container_ID(string): the ID of docker container
    """
    command = f"opensim-cmd run-tool {setup_file_path}"
    output = run_command_onDocker(command, container_ID = container_ID)
    
    return output

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

def run_python_onDocker(source, container_ID = "a8f265a7dadb"):
    environment_info = {
        "LD_LIBRARY_PATH" : [
            "/tmp/moco_dependencies_install/simbody/lib",
            "/tmp/opensim-moco-install/sdk/Python/opensim",
            "/tmp/moco_dependencies_install/adol-c/lib64",
            "/tmp/opensim-moco-install/sdk/Simbody/li",
            ],
    }
    
    command = f"python3.6 -c {repr(source)}"
    output = run_command_onDocker(command, environment_info = environment_info, container_ID = container_ID)
    return output
    
if __name__=="__main__":
    output = run_staticOptimization(template_path = staticOptimization_template_path,
                                    model_path = model_path,
                                    setup_file_path = "/mnt/sdb2/DeepDraw/Projects/20220801_DP02_mri/Opensim/test/setup/test.xml",
                                    ik_path = ik_path,
                                    so_output_dir_path = "/mnt/sdb2/DeepDraw/Projects/20220801_DP02_mri/Opensim/test/so_new",
                                    time_range = (0,5))
    
    load_trc_file("/mnt/sdb2/DeepDraw/Projects/20220801_DP02_mri/Opensim/TRC_files/resnet_50/trial/trial2.trc")

    