
import docker

def run_command_onDocker(command,
                         docker_name = "seojin_opensim"):
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
        print("Container is running. Ready to execute commands.")
    else:
        print("Container is not running. Starting container.")
        container.start()

    # Execute command on docker
    command = f"/bin/bash -c '{command}'"
    exec_result = container.exec_run(cmd = command, tty = True)

    # Output log
    output = exec_result.output.decode('utf-8')

    return output
