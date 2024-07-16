# Title: check_build.py
# Author: Charles "Chuck" Garcai
# Date: 7.11.24
# Description: Verifies correctness of the build stage 

import os, math
import os, logging
import argparse
from ..cutqc.main import CutQC 
from ..cutqc.main import load_cutqc_obj
from ..helper_functions.benchmarks import generate_circ

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


if __name__ == "__main__":
    # Retrieve Pickle Filename
    parser = argparse.ArgumentParser ()
    parser.add_argument ('--filename', default="cutqc_data.pkl", type=str, 
                         help='Filename of Pickled CutQC Instance')
    args = parser.parse_args ()
    
    # Load Pickled CutQC instance
    print ("-- Loading CutQC instance from {} -- ".format (args.filename))
    cutqc = load_cutqc_obj (args.filename)
    
    # Initiate Reconstruction and Verify Results
    print ("-- Building Now --")
    cutqc.build(mem_limit=32, recursion_depth=1)
    cutqc.verify ()
    # cutqc.clean_data()
