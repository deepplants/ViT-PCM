# container

Scripts to help instantiate docker containers with GPU acceleration and python deep learning frameworks. 

## 1. Install Docker
Run:
```bash
bash setup/dockerinstall.sh 
```

Give to docker sudo permissions:
```bash
bash setup/dockersudo.sh
```

To get docker with GPU support run (nvidia-docker2):
```bash
bash setup/nvidiadocker.sh
```
For any issue regarding nvidia-docker please refer to the [official page](https://github.com/NVIDIA/nvidia-docker).

## 2. Build and run containers 
To get custom container running on your home directory:
```bash
bash build.sh -n ContainerNameYouPrefer -f DockerfileYouDefined -i DeepLearningFrameworkImageName -r RequirementsFileName
bash run.sh -n ContainerNameYouPrefer 
```
Please notice that RequirementsFileName must be a path in the current docker context (current build folder - cd ./).

## Examples
To get container with latest Tensorflow and GPU support running on your home directory:
```bash
bash main.sh 
```

## Use different frameworks version

Visit [docker hub](https://hub.docker.com/), search for the framework and select the tag corresponding the version you need (e.g. visit [tensorflow](https://hub.docker.com/r/tensorflow/tensorflow/tags) of [pytorch](https://hub.docker.com/r/pytorch/pytorch/tags)).
Then replace the name in the DeepLearningFrameworkImageName, e.g. if you want to pass to *2.8.0-gpu-jupyter* version, change this:
```bash 
bash build.sh -i tensorflow/tensorflow:latest-gpu-jupyter 
```
into this:
```bash 
bash build.sh -i tensorflow/tensorflow:2.8.0-gpu-jupyter 
```

## How to use in projects
You can define a requirements.txt file in YourProjectFolder and run:
```bash 
cd YourProjectFolder/
bash container/main.sh # this will automatically search for requirements.txt file in the current docker context (YourProjectFolder)
```
