'''General helper functions'''

import os

def get_max_version(directory, filename_prefix, extension):
    """
    Get the maximum version number for a given filename prefix.
    """
    max_version = -1
    for filename in os.listdir(directory):
        if filename.startswith(f"{filename_prefix}_") and filename.endswith(f".{extension}"):
            try:
                version_str = filename.replace(f"{filename_prefix}_", "").replace(f".{extension}", "")
                version_num = int(version_str)
                max_version = max(max_version, version_num)
            except ValueError:
                continue
    return max_version

def generate_filename(directory, filename_prefix, extension):
    """
    Generate an incremental filename for files.
    
    Args:
        directory (str): Directory to check for existing files
        filename_prefix (str): Prefix for the filename
        extension (str): Extension for the filename
    
    Returns:
        str: Next available filename in format "{filename_prefix}_X.{extension}"
    """
    max_version = get_max_version(directory, filename_prefix, extension)
    
    # Generate next version number
    next_version = max_version + 1
    return f"{filename_prefix}_{next_version}.{extension}"

def get_max_filename(directory, filename_prefix, extension):
    """
    Get the maximum filename from a list of filenames.
    """
    max_version = get_max_version(directory, filename_prefix, extension)
    return f"{filename_prefix}_{max_version}.{extension}"
