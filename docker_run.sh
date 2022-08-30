#!/bin/bash
image_name="verify_for_si2022"
tag_name="docker"
script_dir=$(cd $(dirname $0); pwd)

xhost +
docker run -it \
    --net="host" \
    --gpus all \
    --privileged \
    --shm-size=16g \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --name="verify_for_si2022" \
    --volume="$script_dir/:/home/pycode/verify_for_SI2022/" \
    --volume="/media/amsl/96fde31e-3b9b-4160-8d8a-a4b913579ca21/:/home/strage/" \
    $image_name:$tag_name