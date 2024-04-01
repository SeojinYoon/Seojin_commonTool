
import os
from importlib import import_module

def get_library_path(library_name):
    """
    Get library path
    
    :param library_name: the name of library to search
    
    return path of the library(string)
    """
    try:
        # Attempt to import the library/module
        library = import_module(library_name)
        # Get the path to the library's __init__.py file or the .py file for single-file modules
        library_path = library.__file__
        # Return the directory containing the library
        return os.path.dirname(library_path)
    except ModuleNotFoundError:
        return f"Library '{library_name}' is not installed."

if __name__ == "__main__":
    get_library_path("os")

