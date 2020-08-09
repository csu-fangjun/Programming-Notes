#!/bin/bash

if [ $# -ne 1 ]; then
  echo "please provide 1 argument"
  exit 0
fi

name=$(basename -s .c $1)
name=$(basename -s .cc ${name})
make run-$name
