# Standard library imports
import subprocess
import os

# Third party imports
import numpy as np

cached_pt_path = None

# find cornellbox executable 
def find_cornellbox_tracer(starting_path="/"):
    global cached_pt_path
    if cached_pt_path is None:
        name = "CornellBox"
        for root, dirs, files in os.walk(starting_path):
            if name in files:
                cached_pt_path = os.path.join(root, name)
                break
    if cached_pt_path is None:
        raise FileNotFoundError()
    return cached_pt_path

# generate path traced images
def generate_cornell_hemisphere(sample_count = '1024', raydepth = '50', x = '32', y = '32'):
    # search recursively for c++ exec starting one directory up
    starting_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../"
    )
    
    
    proc = subprocess.Popen(
        [ find_cornellbox_tracer(starting_dir),
            '-hemisphere',#raw_filename,
            '-samplecount',
            sample_count,#str(width), (or just 1024?)
            '-raydepth',
            raydepth,#str(height),
            '-x',
            x, #"1",
            '-y',
            y, #"1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = proc.communicate()
    #proc.wait()
    return 1
#generate_cornell_hemisphere()
