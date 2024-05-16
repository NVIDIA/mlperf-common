#!/usr/bin/env python3
import sys
import os
import json
import subprocess
import argparse


def du(path):
    return subprocess.check_output(['du','-sk', path]).split()[0].decode('utf-8')

def get_mount_info(paths_to_verify):
    
    def get_info_records(dir_path):
        records = []
        for root, dirs, files in os.walk(dir_path):
            files = [f for f in files if not f[0] == '.']
            dirs[:] = [d for d in dirs if not d[0] == '.']
            records.append(
                { 
                    "path": root, 
                    "elements": len(files) + len(dirs),
                    "dir_size": int(du(root))
                }
            )
        records.sort(key=lambda x: x['path'], reverse=True)
        return records

    for dir_path in paths_to_verify:
        if not os.path.exists(dir_path):
            print("Error: path $dir_path is incorrect")
            sys.exit(1)

    cont_mount_info = []
    for dir_path in paths_to_verify:
        records = get_info_records(dir_path)
        cont_mount_info.append(
            {
                "name": dir_path,
                "subdirs": records,
            }
        )
        
    cont_mount_info_json = json.dumps(cont_mount_info, indent=4)
    print(cont_mount_info_json)

def verify_mount(cont_mount_info):

    for cont_mount_info_record in cont_mount_info:
        path_to_verify=cont_mount_info_record["name"]
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
                    f"Incorrect number of elements in {path_gt}. "
                    f"Expected {elements_gt}, but found {elements}.")
                break
        
            dir_size_gt = record["dir_size"]
            dir_size = int(du(full_path))
            if abs(dir_size_gt - dir_size) > dir_size_gt * 1e-4:
                print(f"Error: {path_to_verify} is incorrect. "
                    f"Incorrect size of {path_gt}. "
                    f"Expected {dir_size_gt} kB, but is {dir_size} kB.")
                break
            if abs(dir_size_gt - dir_size) > dir_size_gt * 1e-5:
                print(f"Warning: {path_to_verify} may be incorrect. "
                    f"Incorrect size of {path_gt}. "
                    f"Expected {dir_size_gt} kB, but is {dir_size} kB.")
    print(f'Verification completed. See above for all warnings and errors.')
                
def main():
    parser = argparse.ArgumentParser(description='Mount checker')

    parser.add_argument('--check', action='store_true', help='Checking mode')
    parser.add_argument('path', nargs='*', help='Path to JSON file')

    args = parser.parse_args()

    if args.check:
        if len(args.path) == 1:
            with open(args.path[0], 'r') as config_file_read:
                cont_mount_info = json.load(config_file_read)
        elif len(args.path) == 0:
            cont_mount_info = json.load(sys.stdin)
        else:
            raise Exception("Single PATH or no PATH is required in checking mode")
        verify_mount(cont_mount_info)
    else:
        if len(args.path) == 0:
            raise Exception("PATH is required in print mode")
        get_mount_info(args.path)


if __name__ == '__main__':
    main()