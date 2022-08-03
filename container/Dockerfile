# it is possible to use custom args using the flag:
#     docker build . --build-arg some_variable_name=a_value
# and use in the Dockerfile the keyword
#     ARG some_variable_name
# it's possible also to define a default value
#     ARG some_variable_name=b_value

ARG framework_image="tensorflow/tensorflow:latest-gpu"

FROM ${framework_image}

RUN mkdir /workdir
WORKDIR /workdir

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Rome

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests htop tmux \
      python3 python3-pip python3-tk curl git-all nano ffmpeg libsm6 libxext6 \
      libncurses5-dev libprotobuf-dev make cmake protobuf-compiler g++ apt-utils zip unzip wget -y

RUN cd /tmp && git clone https://github.com/Syllo/nvtop.git && mkdir -p nvtop/build && cd nvtop/build && cmake .. && make && make install

ARG requirements_path="./requirements.txt"

COPY ${requirements_path} /tmp/pip-tmp/

RUN pip --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/$(basename -- ${requirements_path}) && rm -rf /tmp/pip-tmp


