import os
import random

import sys


def get_file_paths(base_paths, file_format, limit=None):
    file_paths = []

    for base_path in base_paths:
        for root, dirs, files in os.walk(base_path):
            files_processed = 0
            for file_name in files:
                if file_name.endswith(f'.{file_format}'):
                    file_path = os.path.join(root, file_name)
                    file_paths.append(file_path)
                    files_processed += 1

    random.shuffle(file_paths)
    return file_paths[:limit if limit is not None else sys.maxsize]
