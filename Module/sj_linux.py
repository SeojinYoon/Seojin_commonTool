
import os
from pathlib import Path
import glob 
import subprocess

def exec_command(command, 
                 parameter_info = {}, 
                 pipeline_info = {},
                 is_print = True):
    """
    Execute command
    
    :param command: command
    :param parameter_info(dictionary): argument information
        -k [argument_name]: [argument_value]
        -k 1, [value]: if there is not arg name, set the name using number
    :param pipeline_info(dictionary): pipeline
    :param is_print(boolean): flag for printing command
    
    return: result of command
    """ 
    commands = make_command(command, parameter_info, pipeline_info)
    if is_print:
        print(f"\033[1m" + commands + "\033[0m")
    
    result = os.system(commands)
    return result

def exec_command_inConda(command, 
                         conda_env_name,
                         conda_env_path = "/home/seojin/anaconda3/condabin",
                         parameter_info = {}, 
                         pipeline_info = {},
                         is_print = True):
    """
    Execute command in conda environment
    
    :param command: command
    :param parameter_info(dictionary): argument information
        -k [argument_name]: [argument_value]
        -k 1, [value]: if there is not arg name, set the name using number
    :param pipeline_info(dictionary): pipeline
    :param is_print(boolean): flag for printing command
    
    return: result of command
    """ 
    os.environ["PATH"] += f":{conda_env_path}"
    
    active_conda_command = f"conda run -n {conda_env_name}"
    origin_command = make_command(command, parameter_info, pipeline_info)
    
    commands = " ".join([active_conda_command, origin_command])
    if is_print:
        print(f"\033[1m" + commands + "\033[0m")
    
    result = os.system(commands)
    return result

def exec_command_withSudo(command, 
                          password, 
                          parameter_info = {}, 
                          pipeline_info = {},
                          is_print = True):
    """
    Execute command on sudo
    
    :param command: command
    :param password(string): password of current account
    :param parameter_info(dictionary): argument information
        -k [argument_name]: [argument_value]
        -k 1, [value]: if there is not arg name, set the name using number
    :param pipeline_info(dictionary): pipeline
        -k >: redirect
        -k >>: append
    :param is_print(boolean): flag for printing command
    
    return: result of command
    """ 
    command = f"echo {password}" + " | " + "sudo -S " + make_command(command)
    
    result = exec_command(command, parameter_info, pipeline_info, is_print)
    return result
    
def make_command(command, 
                 parameter_info = {}, 
                 pipeline_info = {}):
    """
    Make command
    
    :param command: command
    :param parameter_info(dictionary): argument information
        -k [argument_name]: [argument_value]
        -k 1, [value]: if there is not arg name, then set the name using number
    :param pipeline_info(dictionary): pipeline
    """    
    arg_str = ""
    for argument in parameter_info:
        arg_type = type(argument)
        
        value = parameter_info[argument]
        value_type = type(value)
        
        # Value parsing
        value_str = ""
        if value_type == list:
            if len(value) > 0:
                for e in value:
                    value_str += str(e) + " "
                
        elif value_type == str:
            if value_type != "":
                value_str = value                
        else:
            value_str = str(value)
        
        # Make arg str
        if arg_type == str:
            if value_str == "":
                arg_str += " -" + argument
            else:
                arg_str += " -" + argument + " " + value_str
        elif arg_type == tuple:
            for a, v in zip(argument, value):
                arg_str += " -" + a + " " + v
        else:
            arg_str += " " + value_str
        
    command = command + arg_str
    
    for pipe in pipeline_info:
        command += " " + pipe + " " + pipeline_info[pipe]
    return command

def rename(target_dir_path, from_, to_, max_depth = 1, file_type = "f"):
    """
    Rename file names
    
    :param target_dir_path(string): target directory path
    :param from_(string): 
    :param to_(string): 
    :param max_depth(int):
    :param file_type(string):
    """
    output = exec_command(command = "find", 
                          parameter_info = {
                              1 : target_dir_path,
                              2 : f"-maxdepth {max_depth}",
                              "type" : file_type,
                              "execdir" : f"rename 's/{from_}/{to_}/'",
                              3 : "{} \;",
                          })
    return output

def make_export_command(environment_info = {}):
    """
    Make export command
    
    :param environment_info(dictionary): ex)
        {
            "LD_LIBRARY_PATH" : [
                "/tmp/moco_dependencies_install/simbody/lib",
                "/tmp/opensim-moco-install/sdk/Python/opensim",
                "/tmp/moco_dependencies_install/adol-c/lib64",
                "/tmp/opensim-moco-install/sdk/Simbody/li",
                ],
        }
    
    return (string): command to export environment variable
    """
    exports = []
    for environment_variable in environment_info:
        values = environment_info[environment_variable]

        for value in values:
            exports.append(f"export {environment_variable}=${environment_variable}:{value}")

    return "; ".join(exports)

if __name__ == "__main__":
    exc_command("ls", {"-l", ""}, {">" : "ls.txt"})
    make_export_command({
        "s" : "s",
    })
    
    exec_command_inConda(command = "ls",
                         conda_env_name = "DP",
                         conda_env_path = "/home/seojin/anaconda3/condabin")