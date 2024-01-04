import os
import logging

from typing import List

def is_existing_file(file_path: str) -> None:
    # Unknwon file
    if not os.path.exists(file_path):
        raise ValueError(f"Unknown file: {file_path}")
    # Umpty file
    if os.path.getsize(file_path) == 0:
        raise ValueError(f"Umpty file: {file_path}")

def open_file_and_give_content(file_path: str) -> List[str]:
    """
    Open a file, read its content, and return the lines as a list of strings.

    Parameters:
    - file_path (str): The path to the file to be read.

    Returns:
    - List[str]: A list containing the lines of the file.

    Raises:
    - ValueError: If the file is unknown (does not exist) or empty.
    """
    is_existing_file(file_path)
    # Base case
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines


def delete_file(file_path: str) -> None:
    """
    Delete a file and log the deletion.

    Parameters:
    - file_path (str): The path to the file to be deleted.

    Raises:
    - ValueError: If the file path is unknown, or if there are permission issues or other exceptions.
    """
    try:
        os.remove(file_path)
        logging.info(f"The file {file_path} has been deleted.")
    except FileNotFoundError:
        raise ValueError(f"The path {file_path} is unknown.")
    except PermissionError:
        raise ValueError(
            f"You don't have the permission to delete the file {file_path}.")
    except Exception as e:
        raise ValueError(f"A problem occured: {e}")


def is_good_type(value, intended_type):
    """
    Check if the type of a value matches the intended type.

    Parameters:
    - value: The value to check.
    - intended_type: The intended type.

    Raises:
    - ValueError: If the type of the value does not match the intended type.
    """
    if type(value) != intended_type:
        raise ValueError(
            f"Incorrect type for {value} : expected {intended_type} and given {type(value)}")


if __name__ == '__main__':
    is_good_type(7, int)
    print(open_file_and_give_content("realsense/utils/processing_general.py"))
    open_file_and_give_content("?")
