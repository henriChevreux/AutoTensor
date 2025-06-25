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

def get_sorted_files_by_version(filepaths, filename_prefix, extension):
    """
    Get a list of filepaths sorted by their version number.
    
    Args:
        filepaths (list): List of file paths to sort
        filename_prefix (str): Prefix for the filename (e.g., 'analysis', 'model')
        extension (str): Extension for the filename (e.g., 'txt', 'py')
    
    Returns:
        list: List of filepaths sorted by version number (oldest to newest)
    """
    files = []
    for filepath in filepaths:
        filename = os.path.basename(filepath)  # Extract just the filename from the path
        if filename.startswith(f"{filename_prefix}_") and (filename.endswith(f".{extension}") if extension else True):
            try:
                version_str = filename.replace(f"{filename_prefix}_", "").replace(f".{extension}", "")
                version_num = int(version_str)
                files.append((version_num, filepath))
            except ValueError:
                continue
    
    # Sort by version number and return just the filepaths
    files.sort(key=lambda x: x[0])
    return [filepath for _, filepath in files]
