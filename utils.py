from pathlib import Path
from tkinter import Tk
from tkinter import filedialog

def get_file_dialog(initial_dir="/media/georg/htcondor/shared-paton/georg/DAtime/data", verbose=True):
    root = Tk()         # create the Tkinter widget
    root.withdraw()     # hide the Tkinter root window

    # Windows specific; forces the window to appear in front
    # root.attributes("-topmost", True)

    path = Path(filedialog.askopenfilename(initialdir=initial_dir, title="Select file"))

    root.destroy()

    if verbose:
        print("selected path: ", path)

    return path


def get_dir_dialog(initial_dir="/media/georg/htcondor/shared-paton/georg/DAtime/data", verbose=True):
    
    root = Tk()         # create the Tkinter widget
    root.withdraw()     # hide the Tkinter root window

    # Windows specific; forces the window to appear in front
    # root.attributes("-topmost", True)

    path = Path(filedialog.askdirectory(initialdir=initial_dir, title="Select data directory"))

    root.destroy()

    if verbose:
        print("selected path: ", path)

    return path


