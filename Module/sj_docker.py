
import docker
from sj_linux import make_export_command

def run_command_onDocker(command,
                         environment_info = {},
                         docker_name = "seojin_opensim2"):
    """
    Run command on docker container

    :param command(string): command
    :param docker_name(string): the name of docker

    return output
    """
    # Get docker container
    client = docker.from_env()
    container = client.containers.get(docker_name)

    # Check whether the container exists
    if container.status == 'running':
        pass
        # print("Container is running. Ready to execute commands.")
    else:
        print("Container is not running. Starting container.")
        container.start()
    
    # Set environment variable
    export_command = make_export_command(environment_info)
    
    # Command for bash
    if export_command != "":
        bash_command = f"/bin/bash -c '{export_command}; {command}'"
    else:
        bash_command = f"/bin/bash -c '{command}'"
    
    # Execute
    print(f"\033[1m" + bash_command + "\033[0m")
    exec_result = container.exec_run(cmd = bash_command, tty = True)

    # Output log
    output = exec_result.output.decode('utf-8')

    return output

if __name__ == "__main__":
    environment_info = {
        "LD_LIBRARY_PATH" : [
            "/tmp/moco_dependencies_install/simbody/lib",
            "/tmp/opensim-moco-install/sdk/Python/opensim",
            "/tmp/moco_dependencies_install/adol-c/lib64",
            "/tmp/opensim-moco-install/sdk/Simbody/li",
            ],
    }

    python_source = f"""
    import os
    import sys
    import opensim as osim

    # Load the model
    model = osim.Model(\\"{model_path}\\")
    """

    command = f'python3.6 -c "{python_source}"'
    s = run_command_onDocker(command, environment_info = environment_info)
    
    