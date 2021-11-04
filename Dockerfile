
FROM pytorch/pytorch:latest

USER root



WORKDIR /home/project

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Madrid
ENV PIP_VERSION="21.1.1"
ENV NODEJS_VERSION=16
ENV JUPYTER_TOKEN=lebasictoken
ENV PORT_JUPYTER=8888
ENV PORT_TENSORBOARD=8887


RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && apt-get install --no-install-recommends -y \
    curl screen vim git locate && \
    apt-get install -y software-properties-common && \
    pip install \
    jupyterlab \
    fire \
    pytorch-lightning \
    scipy \
    matplotlib \
    imageio \
    pyro-ppl && \
    rm -rf /var/lib/apt/lists/*
