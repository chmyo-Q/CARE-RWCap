FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda-12.6 \
    PATH=/opt/care/bin:/usr/local/cuda-12.6/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential python3.12-venv python3.12-dev libprotobuf32t64 \
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
        -o /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb && apt-get update \
    && apt-get install -y --no-install-recommends cuda-toolkit-12-6 \
        libnvinfer10=10.7.0.23-1+cuda12.6 \
        libnvinfer-plugin10=10.7.0.23-1+cuda12.6 \
        libnvonnxparsers10=10.7.0.23-1+cuda12.6 \
    && python3.12 -m venv /opt/care \
    && rm -rf /var/lib/apt/lists/* /tmp/cuda-keyring.deb

WORKDIR /opt/CARE-RWCap
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python scripts/check_environment.py --build-only && python scripts/build.py
CMD ["bash"]
