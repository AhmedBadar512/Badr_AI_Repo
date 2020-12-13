#!/bin/bash
# DOCKERNAME = nvcr.io/navinfo/aicv/horovod_fixed:tf2
DOCKERNAME=$1
docker build -t $DOCKERNAME .
echo $DOCKERNAME
xhost +
nvidia-docker run -it --shm-size 4g --rm --net=host --ipc=host -e DISPLAY=$DISPLAY -v /volumes1/:/volumes1/ -v /volumes2/:/volumes2/ -v /data/:/data/ $DOCKERNAME
