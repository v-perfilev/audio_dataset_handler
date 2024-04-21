import os
import random
import sys


def get_file_paths(base_paths, file_format, limit=None):
    """Gather paths to files matching a specific format from given directories, optionally limited by count."""
    file_paths = []  # List to store matching file paths.

    # Walk through each base directory to find files.
    for base_path in base_paths:
        for root, dirs, files in os.walk(base_path):
            # Filter and collect files matching the specified format.
            for file_name in files:
                if file_name.endswith(f'.{file_format}'):
                    file_paths.append(os.path.join(root, file_name))

    # Shuffle the list to mix file order.
    random.shuffle(file_paths)

    # Return the whole list or a limited subset of file paths.
    return file_paths[:limit if limit is not None else sys.maxsize]
