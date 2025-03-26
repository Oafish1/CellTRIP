import argparse
import atexit
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
    '-c', '--cpu', action='store_true', help='Use only CPU')
parser.add_argument(
    '-t', '--timeout', type=int, help='Reservation time before the program exits, default unlimited for workers')
parser.add_argument(
    '-s', '--separate', action='store_true', help='Start a unique ray instance for each GPU')
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
gpu_ids = list(map(lambda x: [int(x)], args.gpus.split(','))) if args.gpus is not None else [[i] for i in range(torch.cuda.device_count())]
if not args.separate: gpu_ids = [sum(gpu_ids, [])]
if len(gpu_ids) == 0 or args.cpu: gpu_ids =[None]

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
    # if i+1==len(gpu_ids) and block: cmd += ' --block'

    # Add resources
    if gpu_id is not None:
        cmd += f' --num-gpus={len(gpu_id)}'
        cmd += f' --resources=\'{{"VRAM": {sum([torch.cuda.get_device_properties(gid).total_memory for gid in gpu_id])}}}\''
        cmd = f'CUDA_VISIBLE_DEVICES={",".join(map(str, gpu_id))} ' + cmd
    else:
        cmd += f' --num-gpus=0'

    # Execute
    print(cmd)
    os.system(cmd)

# Execute timeout
atexit.register(lambda: os.system('ray stop'))
if block:
    while True: time.sleep(60)
elif args.timeout is not None:
    time.sleep(args.timeout)
