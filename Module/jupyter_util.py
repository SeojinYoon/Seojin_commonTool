
from IPython.display import HTML
from base64 import b64encode

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
 
def show_video(video_path, video_width = 400):
    video_file = open(video_path, "r+b").read()
 
    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
    return HTML(f"""<video autoplay width={video_width} controls><source src="{video_url}"></video>""")
