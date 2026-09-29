FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev build-essential ca-certificates \
    libprotobuf32t64 \
    libnvinfer10=10.7.0.23-1+cuda12.6 \
    libnvinfer-plugin10=10.7.0.23-1+cuda12.6 \
    libnvonnxparsers10=10.7.0.23-1+cuda12.6 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV CUDA_HOME=/usr/local/cuda
WORKDIR /opt/care-rwcap
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python scripts/check_environment.py --build-only \
    && python scripts/build.py \
    && python -m unittest discover -s tests -p 'test_*.py' \
    && python tests/check_models.py
CMD ["python", "scripts/check_environment.py"]
