FROM nvcr.io/nvidia/tensorflow:22.02-tf1-py3
RUN apt update -y \
    && apt install --no-install-recommends -y sudo
RUN pip install -U pip

ARG USERNAME=user
ARG GROUPNAME=user
ARG UID=1000
ARG GID=1000
ARG PASSWORD=user
RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID -G sudo $USERNAME && \
    echo $USERNAME:$PASSWORD | chpasswd && \
    echo "$USERNAME   ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER $USERNAME
WORKDIR /home/$USERNAME/deeplab-custom-dataset
COPY models models
COPY data_generator.py.patch .
USER root
RUN sudo cat data_generator.py.patch | patch -p1
USER $USERNAME
RUN pip install --user tf-slim==1.1.0
RUN pip install --user autopep8
RUN pip install --user yapf
RUN pip install --user bandit
RUN pip install --user flake8
RUN pip install --user mypy
RUN pip install --user pycodestyle
RUN pip install --user pylint==2.4.4
RUN curl https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/ci_build/pylintrc --output ../.pylintrc
