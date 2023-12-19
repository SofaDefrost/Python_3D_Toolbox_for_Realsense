import os
import logging

def open_file_and_give_content(file_path):
    # Unknwon file
    if not os.path.exists(file_path):
        raise ValueError(f"Unknown file: {file_path}")
    # Umpty file
    if os.path.getsize(file_path) == 0:
        raise ValueError(f"Umpty file: {file_path}")
    # Cas de base
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def delete_file(file_path):
    try:
        os.remove(file_path)
        logging.info(f"The file {file_path} has been deleted.")
    except FileNotFoundError:
        raise ValueError(f"The path {file_path} is unknown.")
    except PermissionError:
        raise ValueError(f"You don't have the permission to delete the file {file_path}.")
    except Exception as e:
        raise ValueError(f"A problem occured: {e}")

def is_good_type(value,intended_type):
    if type(value)!=intended_type:
        raise ValueError(f"Incorrect type for {value} : expected {intended_type} and given {type(value)}")
    
if __name__ == '__main__':
    is_good_type(7,int)
    # print(open_file_and_give_content("realsense/utils/file_manager.py"))
    # open_file_and_give_content("Unknownfile")
