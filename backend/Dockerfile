# it is necessary to use NVIDIA-CUDA based images for the use of GPUs
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04

WORKDIR /app

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
	software-properties-common \
	build-essential \
	gcc \
	libgl1 \
	libglib2.0-0 \
	libsm6 \
	libxext6 \
	libxrender1 \
	libxcb1 \
	&& add-apt-repository ppa:deadsnakes/ppa \
	&& apt-get update \
	&& apt-get install -y --no-install-recommends \
	python3.13 \
	python3.13-venv \
	python3.13-dev \
	&& rm -rf /var/lib/apt/lists/*

# create virtual env
RUN python3.13 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
	&& python -m pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]