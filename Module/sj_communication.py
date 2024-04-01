
import os
import numpy as np

def scp_download(server_ip, account_name, server_file_path, local_file_path):
    """
    You must add local computer's ssh key in ~/.ssh/authorized_keys

    reference: https://stackoverflow.com/questions/50096/how-to-pass-password-to-scp
    - (local computer) ssh-keygen -t rsa -C "your_email@youremail.com"
    - (server) copy the content of ~/.ssh/id_rsa.pub and lastly add it to the remote machines ~/.ssh/authorized_keys
    - (server) make sure remote machine have the permissions 0700 for ~./ssh folder and 0600 for ~/.ssh/authorized_keys
    """
    command = f"scp {account_name}@{server_ip}:{server_file_path} {local_file_path}"
    print(command)
    os.system(command)

if __name__ == "__main__":
    for subj_number, cam1_path, cam2_path in zip(subj_numbers, cam1_paths, cam2_paths):
    subj_number = str(subj_number).zfill(2)
    directory_path = f"C:\\Users\\USER\\Desktop\\Videos\\DP{subj_number}"

    cam1_paths = [
            '/mnt/sdb2/DeepDraw/Projects/20220801_DP01_mri/Camera_1/1_01_R_20220801092743.mp4',
            '/mnt/sdb2/DeepDraw/Projects/20220801_DP02_mri/Camera_1/1_01_R_20220801092743.mp4',
            '/mnt/sdb2/DeepDraw/Projects/20220802_DP03_mri/Camera_1/1_01_R_20220802083840.mp4',
            ]
    scp_download(server_ip = "166.104.75.133",
                 account_name = "seojin",
                 server_file_path = cam1_path,
                 local_file_path = directory_path + "\\" + "cam1.mp4")

