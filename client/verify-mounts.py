import sys
import os
import json
from hash import hash_directory

if not len(sys.argv) == 2:
    print("Error: get-mount-info.sh takes 1 argument: 1) paths to verify")
    sys.exit(1)

paths_to_verify = sys.argv[1].split(",")

config_file="cont-mount-info.json"
paths_to_verify = sys.argv[1].split(",")
if not os.path.exists(config_file):
    print(f"{config_file} does not exist")
    sys.exit(1)

with open(config_file, 'r') as config_file_read:
    cont_mount_info = json.load(config_file_read)

    

if len(paths_to_verify) != len(cont_mount_info):
    print(f"Error: "
          f"There are {len(paths_to_verify)} paths ordered to be verified. "
          f"The {config_file} file contains {len(cont_mount_info)}.")

for i, (cont_mount_info_record, path_to_verify) in enumerate(zip(cont_mount_info, paths_to_verify)):
    if cont_mount_info_record["name"]is not None and path_to_verify != cont_mount_info_record["name"]:
        print(f"Warning: The name of path #{i} given for verification "
              f"is different from the one in the {config_file} file. " 
              f"\"{path_to_verify}\" is given. "
              f"In the {config_file} file it is \"{cont_mount_info_record['name']}\".")
        
    for record in cont_mount_info_record["subdirs"]:

        path_gt = record["path"]
        full_path = os.path.join(path_to_verify, path_gt)
        if not os.path.exists(full_path):
            print(f"Error: {path_to_verify} is incorrect. Path {path_gt} is missing")
            print(f"{full_path}")
            break

        elements_gt = record["elements"]
        elements = len([e for e in os.listdir(full_path) if not e.startswith('.')])
        if elements_gt != elements:
            print(f"Error: {path_to_verify} is incorrect. "
                  f"Bad number of elements in {path_gt}. "
                  f"Should be {elements_gt}, but is {elements}.")
            break
       
        dir_size_gt = record["dir_size"]
        dir_size = os.path.getsize(full_path)
        if abs(dir_size_gt - dir_size) > dir_size_gt * 1e-4:
            print(f"Error: {path_to_verify} is incorrect. "
                  f"Bad size of {path_gt}. "
                  f"Should be {dir_size_gt}, but is {dir_size}.")
            break
        if abs(dir_size_gt - dir_size) > dir_size_gt * 1e-5:
            print(f"Warning: {path_to_verify} may be incorrect. "
                  f"Bad size of {path_gt}. "
                  f"Should be {dir_size_gt}, but is {dir_size}.")
            