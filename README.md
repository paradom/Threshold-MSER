## Threshold MSER In Situ Plankton Segmentation

Copyright © 2022 Oregon State University

Dominic W. Daprano  
Sheng Tse Tsai   
Moritz S. Schmid  
Christopher M. Sullivan  
Robert K. Cowen  

Hatfield Marine Science Center  
Center for Qualitative Life Sciences  
Oregon State University  
Corvallis, OR 97331  

This program is distributed WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

This program is distributed under the GNU GPL v 2.0 or later license.

Any User wishing to make commercial use of the Software must contact the authors 
or Oregon State University directly to arrange an appropriate license.
Commercial use includes (1) use of the software for commercial purposes, including 
integrating or incorporating all or part of the source code into a product 
for sale or license by, or on behalf of, User to third parties, or (2) distribution 
of the binary or source code to third parties for use with a commercial 
product sold or licensed by, or on behalf of, User.

------

This tool allows for the segmentation of in situ footage of plankton imagery. The 
segment binary takes video files or images as inputs. The frames are flat-fielded 
using a vertical average and then regions of interest (ROIs) are extracted with 
the Maximal Stable Extrema Regions (MSER) algorithm. ROIs are segmented and their 
information saved (area, major_axis, minor_axis, etc) as the output of the program.

#### Setup

In order to build the project CMake 3.0.0, C++17, OpenCV 4.0.0, and OpenMP are all required.

This project has been successfully compiled with g++ 8.0+ on Ubuntu 20.04 and Centos 8.

To setup, first clone the repository.

```
cd Threshold-MSER
mkdir build
cd build
cmake ../
make
```
Now the binary can be run.

```
./segment --help
```

####  Build Options

There are a few build options that can also be specified.  

OpenMP can be disabled with the WITH_OPENMP=OFF flag in cmake.
```
cmake -DWITH_OPENMP=OFF ../
```

Additionally, the segmentation process can be observed by using the VISUAL_MODE=ON flag in cmake.
*If the project is built with VISUAL_MODE=ON it will automatically set WITH_OPENMP=OFF.*
```
cmake -DVISUAL_MODE=ON ../
```

#### Quick start

The only required parameter of segment is the input parameter. This can either be a path to a 
directory containing video files, a path to a video file, or a path to a folder of images.
```
./segment -i <path/to/dir>
```

The segments that are produced can be controlled through several parameters. Most
notably, the --minium and --maximum paraters control the minimum and 
maximum size of a crop that can be extracted from a frame.
