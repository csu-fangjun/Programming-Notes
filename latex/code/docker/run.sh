#!/bin/bash

if [ 0 -eq 1 ]; then
  docker build -f Dockerfile -t mylatex .
else
  docker run -it \
    -v $HOME:$HOME \
    mylatex
fi
