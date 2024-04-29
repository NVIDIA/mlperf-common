#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: init_config.sh takes 1 or 2 arguments: 1) paths to verify 2) path to file with mount check info"
    exit 1
fi

if [ -n "$2" ]; then
    config_file="$2"
else
    config_file="cont-mount-info.sh"
fi

if ! [ -f "$config_file" ]; then
    echo "Error: config file path $config_file is incorrect"
    exit 1
fi

source $config_file

OLD_IFS=$IFS
IFS=","
for dir_path in $1; do
    if ! [ -d "$dir_path" ]; then
        echo "Error: path $dir_path is incorrect"
        exit 1
    fi
done

index=0
directory_sizes_counter=0
directory_sizes_end=0
for dir_path in $1; do

    ((directory_sizes_end += ${number_of_paths_in_dir[$index]}))


    for ((; directory_sizes_counter < directory_sizes_end; directory_sizes_counter++)); do
        subdir_and_size=${directory_sizes[$directory_sizes_counter]}
        read -r subdir_to_check subdir_size_gt <<< "$subdir_and_size"
        subdir_to_check="$dir_path/$subdir_to_check"
        subdir_to_check=${subdir_to_check//\/\//\/} # replace doubled slashes with single slash

        if ! [ -d "$subdir_to_check" ]; then
            echo "Error: $dir_path is incorrectly initialized. Path $subdir_to_check is missing"
            directory_sizes_counter=$directory_sizes_end
            break
        fi

        subdir_size=$(du -s "$subdir_to_check" | cut -f1)

        if [ "$subdir_size_gt" != "$subdir_size" ]; then
            echo "Error: $dir_path is incorrectly initialized. Bad size of $subdir_to_check ($subdir_size_gt and $subdir_size)"
            directory_sizes_counter=$directory_sizes_end
            break
        fi
    done
    ((index++))
done
echo "DIR CHECK OK" # TODO: REMOVE AFTER TESTS 
IFS=$OLD_IFS