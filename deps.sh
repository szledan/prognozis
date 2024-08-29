#! /bin/bash

sudo apt install python3-pip

pip3 install --break-system-packages \
  tensorflow[and-cuda] \
  pandas \
  numpy \
  matplotlib \
  ipython \
  seaborn
