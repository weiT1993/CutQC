# Title: distributed_reconstructor.py
# Author: Charles "Chuck" Garcia

import os
from typing import List
import torch
import torch.distributed as dist

import torch
from cutqc.main import CutQC 
from helper_functions.benchmarks import generate_circ

def example_func ():
    circuit_type = "bv"
    circuit_size = 4
    circuit = generate_circ(
        num_qubits=circuit_size,
        depth=1,
        circuit_type=circuit_type,
        reg_name="q",
        connected_only=True,
        seed=None,
    )
    
    cutqc = CutQC(
        name="%s_%d" % (circuit_type, circuit_size),
        circuit=circuit,
        cutter_constraints={
            "max_subcircuit_width": 10,
            "max_subcircuit_cuts": 10,
            "subcircuit_size_imbalance": 2,
            "max_cuts": 10,
            "num_subcircuits": [2, 3],
        },
        verbose=False,
        load_data=None,
        parallel_reconstruction=None
    )
    
    print ("-- Cut -- ")    
    cutqc.cut()
    if not cutqc.has_solution:
        raise Exception("The input circuit and constraints have no viable cuts")
    print ("-- Done Cutting -- \n")    
    
    print ("-- Evaluate --")
    cutqc.evaluate(eval_mode="sv", num_shots_fn=None)
    print ("-- Done Evaluating -- \n")

    print ("-- Build --")
    cutqc.build(mem_limit=32, recursion_depth=1)
    print ("-- Done Building -- \n")
    
    cutqc.verify ()
    print("Cut: %d recursions." % (cutqc.num_recursions))
    cutqc.clean_data()



# Environment variables set by slurm script
gpus_per_node = int (os.environ["SLURM_GPUS_ON_NODE"])
WORLD_RANK = int(os.environ["SLURM_PROCID"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
LOCAL_RANK = WORLD_RANK - gpus_per_node * (WORLD_RANK // gpus_per_node)
MASTER_RANK = 0
backend = "nccl"
                                
@torch.jit.script
def vec_kronecker(components) -> torch.Tensor:
    res = components[0]
    for kron_prod in components[1:]:
        res = torch.kron(res, kron_prod)
    return res

def single_node(device):
    # -- Represents Computation On a SINGLE node --
    while True:
        # Receive a flag to check if we should continue or exit
        continue_flag = torch.zeros(1, dtype=torch.bool, device=device)
        dist.recv(tensor=continue_flag, src=MASTER_RANK)
        
        if not continue_flag.item():
            print(f"Rank {WORLD_RANK} exiting.")
            break
        
        # Get shape of the batch we are receiving 
        shape_tensor = torch.zeros([3], dtype=torch.int64, device=device) 
        dist.recv(tensor=shape_tensor, src=MASTER_RANK) 
        
        # Create an empty batch tensor and receive its data
        print(f"Rank {WORLD_RANK}, shapetuple = {shape_tensor}")
        batch_received = torch.empty(tuple(shape_tensor), device=device) 
        dist.recv(tensor=batch_received, src=MASTER_RANK)    
      
        # Call vectorized kronecker and do sum reduce on this node
        res = torch.func.vmap(vec_kronecker)(batch_received)
        res = res.sum(dim=0)
        print(f"Res: {res.shape}") 
        
        # Send Back to master
        dist.reduce(res, dst=MASTER_RANK, op=dist.ReduceOp.SUM)

def main(backend):
    print("Creating Distributed Reconstructor")
    print(f"Hello world! This is worker {WORLD_RANK} speaking. I have {WORLD_SIZE - 1} siblings!")
  
    # Set Device 
    device = torch.device(f"cuda:{LOCAL_RANK}")
    torch.cuda.device(device)
  
    # Master Process collect all that needs to be computed
    if WORLD_RANK == MASTER_RANK:
        # Starts master off.
        # Master will repeatedly send tensors to worker and workers must receive each one
        example()
        
        # When master is done, send exit signal to all workers
        exit_flag = torch.zeros(1, dtype=torch.bool, device=device)
        for rank in range(1, WORLD_SIZE):
            dist.send(tensor=exit_flag, dst=rank)
    else:
        single_node(device)    
def init_processes(backend):
    print (f"Called init_processes, backend={backend}")
    dist.init_process_group (backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    print("Exited init_process_group")
    main (backend)

if __name__ == "__main__":
    print(f"args.backend:{backend}")
    print("Local Rank: {}".format(LOCAL_RANK))
    print("World Rank: {}".format(WORLD_RANK))
    print("World Size: {}".format(WORLD_SIZE))
    print ("GPUS-Avail: {}".format (gpus_per_node))
    init_processes(backend=backend)

