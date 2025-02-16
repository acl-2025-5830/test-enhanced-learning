import os
import torch.distributed as dist
import argparse
import socket
import torch.distributed as dist
import json
import copy

def rank0_print(*args, **kwargs):
    """Print, but only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        free_port = s.getsockname()[1]
        print(f"Found free port: {free_port}")

def rank0_print(*args, **kwargs):
    """Print, but only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)

def print_trainable_params(model):
    total = 0
    trainable = 0
    for v in model.parameters():
        total += v.numel()
        if v.requires_grad:
            trainable += v.numel()
    
    rank0_print(f"Total : {total}   Trainable : {trainable}   ({(trainable/total)*100:.2f})%")

def save_args(model_args, data_args, training_args):
    training_args_copy = copy.deepcopy(training_args)
    delattr(training_args_copy, "accelerator_config") 
    delattr(training_args_copy, "distributed_state") 
    delattr(training_args_copy, "__cached__setup_devices") 

    all_args = {
        "model_args": model_args.__dict__,
        "data_args": data_args.__dict__,
        "training_args": training_args_copy.__dict__,
    }
    os.makedirs(training_args.output_dir, exist_ok=True)
    output_path = os.path.join(training_args.output_dir, "args.json")
    with open(output_path, "w") as f:
        json.dump(all_args, f, indent=4)

if __name__=="__main__":
    find_free_port()