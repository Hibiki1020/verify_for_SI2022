#!/bin/bash

image_name='verify_for_si2022'
image_tag='docker'

docker build -t $image_name:$image_tag --no-cache .