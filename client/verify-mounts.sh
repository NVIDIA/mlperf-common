#!/bin/bash
abs() {
    if [ $1 -ge 0 ]; then
        echo $1
    else
        echo $((-$1))
    fi
}

get_similarity() {
    num1=$1
    num2=$2

    percentage_difference=$(( 200 * ($num1 - $num2) / ($num1 + $num2) ))
    percentage_difference=$(abs $percentage_difference)

    echo "$percentage_difference"
    return $percentage_difference
}


if [ "$#" -ne 1 ]; then
    echo "Error: init_config.sh takes 1 argument: 1) paths to verify"
    exit 1
fi

config_file="cont-mount-info.sh"
paths_to_verify=$1
threshold=10

if ! [ -f "$config_file" ]; then
    echo "Error: config file path $config_file is incorrect"
    exit 1
fi

source $config_file

OLD_IFS=$IFS
IFS=","
for dir_path in $paths_to_verify; do
    if ! [ -d "$dir_path" ]; then
        echo "Error: path $dir_path is incorrect"
        exit 1
    fi
done

index=0
directory_sizes_counter=0
directory_sizes_end=0
for dir_path in $paths_to_verify; do

    ((directory_sizes_end += ${number_of_paths_in_dir[$index]}))


    for ((; directory_sizes_counter < directory_sizes_end; directory_sizes_counter++)); do
    echo "$directory_sizes_counter          $subdir_and_size"
        subdir_and_size=${directory_sizes[$directory_sizes_counter]}
        read -r subdir_to_check subdir_size_gt <<< "$subdir_and_size"
        subdir_to_check="$dir_path/$subdir_to_check"
        subdir_to_check=${subdir_to_check//\/\//\/} # replace doubled slashes with single slash

        if ! [ -d "$subdir_to_check" ]; then
            echo "Error: $dir_path is incorrectly initialized. Path $subdir_to_check is missing"
            directory_sizes_counter=$directory_sizes_end
            break
        fi

        subdir_size=$(du -sk "$subdir_to_check" | cut -f1)
        percentage_difference=$(get_similarity "$subdir_size_gt" "$subdir_size")
        if [ "$percentage_difference" -gt "$threshold" ]; then
            echo "Error: $dir_path is incorrectly initialized. Bad size of $subdir_to_check ($subdir_size_gt and $subdir_size) $subdir_and_size"
            directory_sizes_counter=$directory_sizes_end
            break
        fi
    done
    ((index++))
done
echo "DIR CHECK OK" # TODO: REMOVE AFTER TESTS 
IFS=$OLD_IFS