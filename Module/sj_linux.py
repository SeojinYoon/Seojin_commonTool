
import os
from pathlib import Path
import glob 
from sj_string import str_join

def exec_command(command, parameter_info = {}, pipeline_info = {}):
    """
    Execute command
    
    :param command: command
    :param parameter_info(dictionary): argument information
        -k [argument_name]: [argument_value]
        -k 1, [value]: if there is not arg name, set the name using number
    :param pipeline_info(dictionary): pipeline
    """ 
    commands = make_command(command, parameter_info, pipeline_info)
    print(f"\033[1m" + commands + "\033[0m")
    
    result = os.system(commands)
    return result

def make_command(command, parameter_info = {}, pipeline_info = {}):
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
        elif arg_type == list:
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

if __name__ == "__main__":
    exc_command("ls", {"-l", ""}, {">" : "ls.txt"})
    