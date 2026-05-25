
# Common Libraries
import docker

# Custom Libraries
from sj_linux import make_export_command

def run_command_onDocker(command,
                         container_ID,
                         environment_info = {}):
    """
    Run command on docker container

    :param command(string): command
    :param container_ID(string): the ID of docker container

    return output
    """
    # Get docker container
    client = docker.from_env()
    container = client.containers.get(container_ID)

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
    full_script = f"{export_command}; {command}" if export_command else command
    bash_command = ["/bin/bash", "-c", full_script]

    # Execute
    # print(f"\033[1m" + bash_command + "\033[0m")
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
    
    