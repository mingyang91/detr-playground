FROM nvidia/cuda:12.3.1-devel-ubuntu22.04 as builder
RUN apt update && apt install python3.10 python3-pip -y
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN apt install git -y
RUN pip install flash-attn -i https://pypi.tuna.tsinghua.edu.cn/simple
