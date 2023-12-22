# RealSense

This repository provides a set of Python scripts useful for 3D file processing, particularly for files obtained using Realsense depth cameras. It includes an *acquisition_realsense.py* file for capturing and recording with a Realsense camera, as well as a *Utils* folder containing a collection of processing functions (for *.ply* files, arrays or images).

## Usage

To test the code, execute the Python script example.py:
```console
python3 example.py
```
All outcomes will be stored in the 'example' folder.
## Prerequisites
Python libraries required for the entire repository:

```console
pip3 install numpy
pip3 install matplotlib
pip3 install opencv-python
pip3 install pyrealsense2
pip3 install pyvista
pip3 install open3d
```

For additional information about Realsense with Python, visit: https://dev.intelrealsense.com/docs/python2

Authors: Thibaud Piccinali, Tinhinane Smail

Supervision: Paul Chaillou

Revision: Damien Marchal




