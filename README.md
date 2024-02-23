# Python 3D Toolbox for Realsense

This repository provides a set of Python scripts useful for 3D file processing, particularly for files obtained using Realsense depth cameras. It includes an *acquisition_realsense.py* file for capturing and recording with a Realsense camera, as well as a *functions* folder containing a collection of processing functions (point cloud, ply, pixels...) and a subfolder *utils* containing some other useful functions.

Thanks to this repository you should be able to:

- Connect multiple realsense's cameras to one computer and process acquisitions.
- Filter point cloud (density, threshold, radius, ...) thanks (or not) to users interface. 
- Process point cloud (resize, center on image, color, ...).
- Cut point cloud by selecting a zone with an interface.
- Find the homography between two images.
- Determine and apply an HSV mask to a point cloud.
- ...

## Usage

To test the code, execute the Python script example.py:
```console
python3 example.py
```
All outcomes will be stored in the 'example/output' folder.

## Prerequisites

This repository has been created using Python 3.8.10. Using another version may result in some problems. 

Python libraries required for the entire repository:

```console
pip3 install numpy
pip3 install pymeshlab
pip3 install scipy
pip3 install matplotlib
pip3 install opencv-python
pip3 install pyrealsense2
pip3 install pyvista
pip3 install open3d
```

## Important notes

- Please be careful of the shape of lists of colors when working with point clouds. Indeed, if this list is from a camera's acquisition it would be 2D-shaped according to the parameters of the camera (for example (480,640,3)) but otherwise if this list is get from a *.ply* file it would be 1D-shaped (in our example (307200,3)). You can switch between the two shapes by using the functions in *functions/utils/array.py* : *to_line* and *line_to_2Darray*. Most of the functions developed in this repository have been written with respect to this property : if the list of colors is not 2D-shaped you can specify the shape expected as an argument of the function.

- Please be careful to the type of lists of colors when working with images. Indeed all this repository has been built for RGB's images but because of OpenCV (one of the libraries used which is working with BGR's images) you might encounter problems of colors : red and blue pixels could be exchanged. If you are facing this problem you can switch between an RGB and an BGR list by using the code ```[:, :, ::-1]```. For example if ```colors``` is a RGB list, ```colors[:, :, ::-1]``` will be an BGR list (and vice versa).

For additional information about Realsense with Python, visit: https://dev.intelrealsense.com/docs/python2

Authors: Thibaud Piccinali, Tinhinane Smail

Supervision: Paul Chaillou

Revision: Damien Marchal




