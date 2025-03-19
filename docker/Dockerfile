# Make sure to set build context to CellTRIP main folder
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
COPY examples/train.py .
COPY docker/run_node.py .
COPY docker/host_test.py .

# Installing CUDA capability (maybe not needed?)
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

# Do this for running multiple https://stackoverflow.com/a/28490909

# Build
# sudo docker build -t celltrip -f docker/Dockerfile .

# Run host and script
# NOTE: -u unbuffered output enables realtime printing
# --mount type=bind,source=$PWD/output,target=/output \
# sudo docker run \
#     --mount type=bind,source=$PWD/data,target=/data,readonly \
#     --gpus all \
#     --shm-size 5Gb \
#     --net host \
#     celltrip \
#     /bin/bash -c "python run_node.py && python -u train.py"

# Run host with no timeout
# NOTE: -u unbuffered output enables realtime printing
# --mount type=bind,source=$PWD/output,target=/output \
# sudo docker run \
#     --mount type=bind,source=$PWD/data,target=/data,readonly \
#     --gpus all \
#     --shm-size 5Gb \
#     --net host \
#     celltrip \
#     /bin/bash -c "python run_node.py --timeout -1"

# Run worker with no timeout
# --timeout 5
# sudo docker run \
#     --gpus all \
#     --shm-size 5Gb \
#     --net host \
#     celltrip \
#     /bin/bash -c "python run_node.py --address 127.0.0.1:6379"

# Show all docker containers
# sudo docker ps
