import sys
import os
import json
import subprocess



if not len(sys.argv) == 2:
    print("Error: get-mount-info.sh takes 1 argument: 1) paths to verify")
    sys.exit(1)

paths_to_verify = sys.argv[1].split(",")

for dir_path in paths_to_verify:
    if not os.path.exists(dir_path):
        print("Error: path $dir_path is incorrect")
        sys.exit(1)

def du(path):
    return subprocess.check_output(['du','-sk', path]).split()[0].decode('utf-8')

cont_mount_info = []
for dir_path in paths_to_verify:
    records = []
    for root, dirs, files in os.walk(dir_path):
        files = [f for f in files if not f[0] == '.']
        dirs[:] = [d for d in dirs if not d[0] == '.']
        # print(f'root: {"/".join(root.split("/")[1:])}')
        # print(f"files: {files}")
        # print(f"dirs: {dirs}")

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