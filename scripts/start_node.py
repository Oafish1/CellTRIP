import argparse
import os
import time
import warnings

import torch


# Arguments
parser = argparse.ArgumentParser(
    prog='CellTRIP host/worker running script',
    description='Initializes host and worker nodes for Ray, compatible with CellTRIP')
parser.add_argument(
    '-a', '--address', type=str, help='IP:PORT of the head node. If None, this node becomes a head')
parser.add_argument(
    '-n', '--node-ip-address', type=str, help='Node IP address, generally required for smooth operation')
parser.add_argument(
    '-g', '--gpus', type=str, help='CUDA devices to use, separated by commas')
parser.add_argument(
    '-t', '--timeout', type=int, help='Reservation time before the program exits, default unlimited for workers')
parser.add_argument(
    '-e', '--extras', type=str, help='Extra arguments')
args = parser.parse_args()

# Default
is_head = args.address is None
block = False
if args.timeout is None:
    block = not is_head
elif args.timeout == -1:
    block = True
    args.timeout = None

# Get resources
gpu_ids = list(map(int, args.gpus.split(','))) if args.gpus is not None else [i for i in range(torch.cuda.device_count())]
num_gpus = len(gpu_ids)
if len(gpu_ids) == 0: gpu_ids =[None]
vram = [torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus)]

# Initialize ray
command = 'ray start --disable-usage-stats'
if args.node_ip_address is not None: command += f' --node-ip-address {args.node_ip_address}'
else: warnings.warn('`node_ip_address` not provided, may cause unexpected errors on public/mixed networks')
if args.extras is not None: command += ' ' + args.extras

# Run command
for i, gpu_id in enumerate(gpu_ids):
    cmd = command

    # Add head
    if is_head and i+1==len(gpu_ids): cmd += ' --head --port=6379 --dashboard-host=0.0.0.0'
    elif is_head: cmd += f' --address 127.0.0.1:6379'
    else: cmd += f' --address {args.address}'

    # Block if last
    if i+1==len(gpu_ids) and block: cmd += ' --block'

    # Add resources
    if gpu_id is not None:
        cmd += f' --num-gpus=1'
        cmd += f' --resources=\'{{"VRAM": {torch.cuda.get_device_properties(gpu_id).total_memory}}}\''
        cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} ' + cmd

    # Execute
    print(cmd)
    os.system(cmd)

# Execute timeout
if args.timeout is not None:
    time.sleep(args.timeout)
    os.execute('ray stop')
