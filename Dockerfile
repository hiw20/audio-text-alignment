# FROM ubuntu:20.04
# FROM anibali/pytorch:2.0.0-cuda11.8-ubuntu22.04
# USER root
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip ffmpeg
RUN python3 -m pip install --no-cache-dir --upgrade pip

# ARG REF=main
# RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF
# RUN python3 -m pip install --no-cache-dir -e ./transformers[dev-torch,testing,video]

# If set to nothing, will install the latest version
ARG PYTORCH='2.0.0'
ARG TORCH_VISION=''
ARG TORCH_AUDIO=''
# Example: `cu102`, `cu113`, etc.
ARG CUDA='cu117'

RUN [ ${#PYTORCH} -gt 0 ] && VERSION='torch=='$PYTORCH'.*' ||  VERSION='torch'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA
RUN [ ${#TORCH_VISION} -gt 0 ] && VERSION='torchvision=='TORCH_VISION'.*' ||  VERSION='torchvision'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA
RUN [ ${#TORCH_AUDIO} -gt 0 ] && VERSION='torchaudio=='TORCH_AUDIO'.*' ||  VERSION='torchaudio'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA


# RUN sudo apt-get update && apt-get upgrade -y && apt-get clean
# RUN apt install -y software-properties-common 
# RUN add-apt-repository -y ppa:deadsnakes/ppa
# RUN apt install python3.9 -y
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 3
# RUN apt install python3-pip -y

RUN mkdir /app

RUN apt update
RUN apt install git -y
RUN git clone https://github.com/daandouwe/ngram-lm.git /app

COPY ./requirements.txt /app
RUN pip install -r app/requirements.txt

COPY . /app
COPY ngram_data/* ngram-lm

# RUN cd /app && python3 /app/generate_data.py --start=0 --end=1 && python3 /app/generate_metrics.py --start=0 --end=1 && python3 /app/novel_alignment.py
