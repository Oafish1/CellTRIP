# Make sure to set build context to CellTRIP main folder
# NOTE: Must be python version 3.10.16 to interface with containers
FROM rayproject/ray:2.44.0.f468b3-py310-gpu
USER root
WORKDIR /run

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# Clone CellTRIP
COPY celltrip/ celltrip/
COPY setup.py .
COPY README.md .
RUN python -m pip install --no-cache-dir .

# Get runfiles
COPY scripts/train.py .
COPY scripts/start_node.py .

### Installing CUDA capability
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# sudo systemctl restart docker

### Build
# sudo docker build -t celltrip .

### Save to file
# sudo docker save -o celltrip.tar celltrip
# sudo chmod a+rw celltrip.tar
# sudo docker load --input celltrip.tar

### Run host and script
# NOTE: -u unbuffered output enables realtime printing
## Docker
# --mount type=bind,source=$PWD/output,target=/output \
# sudo docker run \
#     --mount type=bind,source=$PWD/data,target=/data,readonly \
#     --gpus all \
#     --shm-size 5Gb \
#     --net host \
#     celltrip \
#     /bin/bash -c "python start_node.py --node-ip-address <LOCAL-IP> && python -u train.py"
## Local
# python scripts/start_node.py --node-ip-address <LOCAL-IP>

### Run host indefinitely
# NOTE: -u unbuffered output enables realtime printing
## Docker
# --mount type=bind,source=$PWD/output,target=/output \
# sudo docker run \
#     --mount type=bind,source=$PWD/data,target=/data,readonly \
#     --gpus all \
#     --shm-size 5Gb \
#     --net host \
#     celltrip \
#     /bin/bash -c 'python start_node.py --node-ip-address <LOCAL-IP> --timeout -1'
## Local
# python scripts/start_node.py --node-ip-address <LOCAL-IP> --timeout -1

### Run worker indefinitely
## Docker
# --timeout 5
# GPU selection: --gpus '"device=1,2"'
# sudo docker run \
#     --gpus all \
#     --shm-size 5Gb \
#     --net host \
#     celltrip \
#     /bin/bash -c "python start_node.py --address <HEAD-IP> --node-ip-address <LOCAL-IP>"
## Local
# python scripts/start_node.py --address <HEAD-IP> --node-ip-address <LOCAL-IP>

# Other useful commands
# sudo docker ps
# sudo docker container ls
# sudo docker container stop <ID>
