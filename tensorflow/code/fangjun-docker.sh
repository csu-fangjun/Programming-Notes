#!/bin/bash


TAG=2020-05-13
docker run -it --rm \
  -v /opt/intel:/opt/intel \
  -v /home/fangjunkuang:/home/fangjunkuang \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=6,7 \
  atp-hotword-ci:$TAG
