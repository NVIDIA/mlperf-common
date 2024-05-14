import sys
import os
import json
import subprocess

def du(path):
    return subprocess.check_output(['du','-sk', path]).split()[0].decode('utf-8')

def getMountInfo(paths_to_verify):

    paths_to_verify = paths_to_verify.split(",")

    for dir_path in paths_to_verify:
        if not os.path.exists(dir_path):
            print("Error: path $dir_path is incorrect")
            sys.exit(1)

    cont_mount_info = []
    for dir_path in paths_to_verify:
        records = []
        for root, dirs, files in os.walk(dir_path):
            files = [f for f in files if not f[0] == '.']
            dirs[:] = [d for d in dirs if not d[0] == '.']

            records.append(
                { 
                    "path": "/".join(root.split("/")[1:]), # root without first dir (a/b/c --> b/c)
                    "elements": len(files) + len(dirs),
                    "dir_size": int(du(root))
                }
            )
        
        records.sort(key=lambda x: x['path'], reverse=True)
        cont_mount_info.append(
            {
                "name": dir_path,
                "subdirs": records,
            }
        )
        
    cont_mount_info_json = json.dumps(cont_mount_info, indent=4)
    print(cont_mount_info_json)

def verifyMount(paths_to_verify):
    config_file="cont-mount-info.json"
    paths_to_verify = paths_to_verify.split(",")
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
            dir_size = int(du(full_path))
            if abs(dir_size_gt - dir_size) > dir_size_gt * 1e-4:
                print(f"Error: {path_to_verify} is incorrect. "
                    f"Bad size of {path_gt}. "
                    f"Should be {dir_size_gt}, but is {dir_size}.")
                break
            if abs(dir_size_gt - dir_size) > dir_size_gt * 1e-5:
                print(f"Warning: {path_to_verify} may be incorrect. "
                    f"Bad size of {path_gt}. "
                    f"Should be {dir_size_gt}, but is {dir_size}.")
                