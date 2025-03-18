import argparse
import os
import time

import torch


# Arguments
parser = argparse.ArgumentParser(
    prog='CellTRIP host/worker running script',
    description='Initializes host and worker nodes for Ray, compatible with CellTRIP')
parser.add_argument(
    '-a', '--address', type=str, help='IP:PORT of the head node. If None, this node becomes a head')
parser.add_argument(
    '-t', '--timeout', type=int, help='Reservation time before the program exits, default unlimited for workers')
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
num_gpus = torch.cuda.device_count()
vram = [torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus)]

# Initialize ray
command = 'ray start --disable-usage-stats'
if block: command += ' --block'
# Host and worker arguments
if is_head: command += ' --head --port=6379 --dashboard-host=0.0.0.0'
else: command += f' --address {args.address}'

# Run command
if num_gpus == 0:
    # Execute
    cmd = command
    os.system(cmd)
for i in range(num_gpus):
    # Add resources
    cmd = command
    cmd += f' --num-gpus=1'
    cmd += f' --resources=\'{{"VRAM": {torch.cuda.get_device_properties(i).total_memory}}}\''
    # Add prefix
    # https://github.com/ray-project/ray/issues/7486#issuecomment-596057031
    cmd = f'CUDA_VISIBLE_DEVICES={i} ' + cmd
    # Execute
    os.system(cmd)

if args.timeout is not None:
    time.sleep(args.timeout)

# CLI Output
# print(cmd)
# import time
# time.sleep(10)
# import ray
# ray.init()
# print(ray.available_resources())
