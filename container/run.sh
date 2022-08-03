#!/bin/bash

name='tf'
container_name="${USER}-${PWD##*/}-${name}"

while getopts n:c: flag
do
    case "${flag}" in
        n) name=${OPTARG};;
        c) container_name=${OPTARG};;
    esac
done

image_name="${name}/${name}"

if [ ! "$(docker ps -q -f name=${container_name})" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=${container_name})" ]; then
        # cleanup
        docker rm ${container_name}
    fi
    # run your container 
    docker run -u $(id -u):$(id -g) --shm-size=64gb --volume="$PWD:/workdir" --volume="/etc/passwd:/etc/passwd:ro" --volume="/etc/shadow:/etc/shadow:ro" --name ${container_name} --rm --gpus all --runtime nvidia --pid host -d -t ${image_name} bash -c "tmux new -s ${name}"
fi
docker exec -it ${container_name} bash -c "tmux set -g window-size largest; tmux attach -t ${name}"



