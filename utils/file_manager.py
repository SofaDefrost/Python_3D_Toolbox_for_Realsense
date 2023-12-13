import os

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


if __name__ == '__main__':
    print(open_file_and_give_content("realsense/utils/file_manager.py"))
    open_file_and_give_content("Unknownfile")