FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu18.04

ENV LC_ALL=C.UTF-8
ENV LANG=c.UTF-8

RUN apt-get update
RUN apt-get -y install python3-pip
RUN pip3 install --upgrade pip

WORKDIR /api

COPY requirements.txt .

RUN pip install -r requirements.txt && \
    pip install uvloop && \
    pip install tensorflow-gpu && \
    pip install pandas && \
    pip install torch

COPY . /api

CMD ["uvicorn", "api.main:app", "--app-dir=./", "--reload", "--workers=1", "--host=0.0.0.0", "--port", "8000", "--use-colors", "--loop=uvloop"]
