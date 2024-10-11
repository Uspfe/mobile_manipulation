
import argparse
import os
import shutil
import yaml
import copy
from mmseq_utils.plotting import construct_logger
import numpy as np


def copy_files(src_folder, new_folder_name):
    # Get the parent directory of the source folder
    parent_folder = os.path.dirname(src_folder)
    
    # Create the destination folder in the parent directory
    dest_folder = os.path.join(parent_folder, new_folder_name)
    
    # If the destination folder doesn't exist, create it
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Walk through the source folder recursively
    for root, dirs, files in os.walk(src_folder):
        # Calculate relative path from src_folder
        rel_path = os.path.relpath(root, src_folder)

        # Create the corresponding folder in the destination
        dest_subfolder = os.path.join(dest_folder, rel_path)
        if not os.path.exists(dest_subfolder):
            os.makedirs(dest_subfolder)

        # Check if the current folder starts with "controller"
        folder_name = os.path.basename(root)
        skip_data_file = folder_name.startswith('controller')

        # Copy files, skip data.npz only if inside a folder starting with "controller"
        for file in files:
            if not (skip_data_file and file == 'data.npz') and not file.endswith("bag"):
                src_file_path = os.path.join(root, file)
                dest_file_path = os.path.join(dest_subfolder, file)
                shutil.copy2(src_file_path, dest_file_path)
                print(f"Copied {src_file_path} to {dest_file_path}")
    
    return dest_folder


def dump_dict_to_yaml(dictionary, folder):
    # Look for a subfolder that starts with "control"

    yaml_path = os.path.join(folder, 'config.yaml')

    # Dump the dictionary into the config.yaml file
    with open(yaml_path, 'w') as yaml_file:
        yaml.safe_dump(dictionary, stream=yaml_file, default_flow_style=False)

    print(f"Dictionary has been written to {yaml_path}")
    return

def list_of_dicts_to_dict_of_arrays(list_of_dicts):
    # Collect all keys from the first dictionary
    keys = list_of_dicts[0].keys()
    
    # Create an empty dictionary to hold the stacked arrays
    dict_of_arrays = {}
    
    # For each key, stack the arrays directly using np.concatenate
    for key in keys:
        # Use tuple comprehension to gather all arrays for the current key
        new_key = "_".join(["mpc", key])+"s"
        dict_of_arrays[new_key] = np.vstack([[d[key]] for d in list_of_dicts])
    
    return dict_of_arrays

def get_control_folder(directory):
    # List all subdirectories in the given directory
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        # Check if the subdirectory name starts with "control" and it's a directory
        if os.path.isdir(subdir_path) and subdir.startswith('control'):
            return subdir_path
    
    return None  # Return None if no matching subdirectory is found

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--folder", required=True, help="Path to data folder.")

    args = parser.parse_args()

    data_plotter = construct_logger(args.folder, process=False)
    
    ctrl_modes = data_plotter.config["controller"]["control_modes"]
    ctrl_mode_names = [d["key"] for d in ctrl_modes]

    # Get the directory name
    dir_name = os.path.basename(os.path.normpath(args.folder))
    controller = data_plotter.controller
    controller_sequence = data_plotter.data.pop("mpc_curr_controllers")
    controller_data = {name: data_plotter.data.pop("mpc_" + name + "s") for name in ctrl_mode_names}
    
    # copy other files to new directories, each for one control mode
    for name in ctrl_mode_names:
        # Separate Config File
        dest_folder_name = "_".join([dir_name, name])
        dest_folder = copy_files(args.folder, dest_folder_name)
        ctrl_mode_config = copy.deepcopy(controller.controllers[name].params)
        ctrl_mode_config = {"controller": ctrl_mode_config}
        control_folder_path = get_control_folder(dest_folder)
        dump_dict_to_yaml(ctrl_mode_config, control_folder_path)

        # Seperate Data
        time_idx = np.where(controller_sequence == name)[0]
        data_new = {}
        for key in data_plotter.data.keys():
            if isinstance(data_plotter.data[key], np.ndarray) and data_plotter.data[key].shape != ():
                if len(data_plotter.data[key]) == len(time_idx):
                    data_new[key] = data_plotter.data[key]
                else:
                    data_new[key] = data_plotter.data[key][time_idx]
            else:
                data_new[key] = data_plotter.data[key]
        data_ctrl = controller_data[name]
        data_ctrl = [data_plotter._convert_np_array_to_dict(data_ctrl[t_index]) for t_index in time_idx]

        data_ctrl = list_of_dicts_to_dict_of_arrays(data_ctrl)
        data_new.update(data_ctrl)

        data_path = os.path.join(control_folder_path, "data.npz")
        np.savez_compressed(data_path, **data_new)


    
    


