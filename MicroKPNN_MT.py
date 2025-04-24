import sys
import os
import subprocess
import argparse

from MicroKPNN_MT_lib.config import config

######################
# ## PARSE ARGUMENTS ###
# ######################
# parser = argparse.ArgumentParser(description='MicroKPNN')

# parser.add_argument('--data_path', type=str, required=True, help='Path to data')
# parser.add_argument('--metadata_path', type=str, required=True, help='Path to metadata')
# parser.add_argument('--device', type=int, default=0, help='Which gpu to use if any (default: 0)')
# parser.add_argument('--output', type=str, help='path to output folder')
# parser.add_argument('--taxonomy', type=str , default=5, help='taxonomy to use for hidden layer')
# parser.add_argument('--k_fold', type=int, default=5, help='k for k-fold validation')
# args = parser.parse_args()

data_cfg = config["data"]

# Create sub directories in output directory
if not os.path.exists("Results/MicrokPNN_MT/NetworkInput"):
    cmd = "mkdir -p Results/MicrokPNN_MT/NetworkInput"
    check_result=subprocess.check_output(cmd, shell=True)
if not os.path.exists("Results/MicrokPNN_MT/Record"):
    cmd = f"mkdir -p Results/MicrokPNN_MT/Record"
    check_result=subprocess.check_output(cmd, shell=True)
if not os.path.exists("Results/MicrokPNN_MT/Checkpoint"):
    cmd = f"mkdir -p Results/MicrokPNN_MT/Checkpoint"
    check_result=subprocess.check_output(cmd, shell=True)
print("Create/Check existance of NetworkInput dir, Record dir and Checkpoint dir in directory ", "Results/MicrokPNN_MT" )

# create specied taxonomy info
cmd = f"python MicroKPNN_MT_lib/taxonomy_info.py --inp {data_cfg['train_abundance_path']} --out Results/MicrokPNN_MT/NetworkInput/"
check_result=subprocess.check_output(cmd, shell=True)
print("create species_info.pkl")

# create edges
cmd = f"python MicroKPNN_MT_lib/create_edges.py --inp {data_cfg['train_abundance_path']} --taxonomy 5 --out Results/MicrokPNN_MT/NetworkInput/"
check_result=subprocess.check_output(cmd, shell=True)
print("EdgeList has been created")

# train model
edges= "Results/MicrokPNN_MT/NetworkInput/EdgeList.csv"


cmd = f"python MicroKPNN_MT_lib/train_meta_kfold.py --k_fold 5 --data_path {data_cfg['train_abundance_path']} --metadata_path {data_cfg['train_metadata_path']} --edge_list {edges} --output Results/MicrokPNN_MT --checkpoint_path Results/MicrokPNN_MT/Checkpoint/microkpnn_mt.pt --records_path Results/MicrokPNN_MT/Record/microkpnn_mt.csv --device 0"
check_result=subprocess.check_output(cmd, shell=True)
print("Well Done")