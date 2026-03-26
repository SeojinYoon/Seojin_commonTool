
def is_jupyter():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter Notebook or Lab
        elif shell == 'TerminalInteractiveShell':
            return False  # IPython terminal
        else:
            return False
    except:
        return False


