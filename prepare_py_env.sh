# This file can be sourced to prepare the python environment. Disregard this if you can alread import cv2 within Python.
# OpenCV was installed by running
# clone <opencv repo>
# cd <opencv repo>
# mkdir build
# cd build
# cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=~/opencv_install -D OPENCV_EXTRA_MODULES_PATH=~/opencv/opencv_contrib/modules -D PYTHON3_EXECUTABLE=/usr/bin/python3 ..
# make install
# Naturally, you should replace /home/vsantos with whatever you want, namely your own username.
CV_DIR="/home/vsantos/opencv_install/lib/python3.5/dist-packages/"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CV_DIR}
export PYTHONPATH=${CV_DIR}:${PYTHONPATH} "$@"