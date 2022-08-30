#!/bin/bash
image_name="verify_for_si2022"
tag_name="docker"
script_dir=$(cd $(dirname $0); pwd)

docker run -it \
    --net="host" \
    --gpus all \
    --privileged \
    --shm-size=400g \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --name="verify_for_si2022" \
    --volume="$script_dir/:/home/pycode/verify_for_SI2022/" \
    --volume="/home/kawai/ssd_dir/:/home/ssd_dir/" \
    --volume="/fs/kawai/:/home/strage/" \
    $image_name:$tag_name