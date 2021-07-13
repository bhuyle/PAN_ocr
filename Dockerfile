# Chọn Base Image để build
FROM nvidia/cuda:10.0-runtime-ubuntu18.04
# FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime
# FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

WORKDIR /home

# Cài đặt python 3.7
RUN apt update && \
    apt upgrade -y && \
    apt install -y libsm6 libxext6 libxrender-dev software-properties-common &&\
    add-apt-repository ppa:deadsnakes/ppa -y && \
    # apt install -y python3-pip && \
    # python3 -m pip install --upgrade pip
    apt install -y python3.7 python3.7-dev python3-pip &&\
    ln -sf /usr/bin/python3.7 /usr/bin/python3

# Copy các file cài đặt cần thiết vào thư mục hiện tại trong image (/home)
COPY requirement.txt .

# Cài đặt các thư viện cần thiết để chạy
RUN pip3 install -r requirement.txt

CMD bash 
