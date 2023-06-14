FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    git
RUN git clone https://github.com/naseemap47/autoAnnoter.git home
WORKDIR /home
RUN pip install triton==2.0.0 super-gradients
RUN pip install ultralytics lxml onnxruntime streamlit wget
RUN pip install transformers addict yapf timm supervision==0.6.0
RUN pip install -e GroundingDINO/