# Face Recognition Using Estimated 3D Facial Landmarks with Stereo Images

This paper proposes a face recognition system that uses3D facial landmarks to recognize the face. These landmarksare estimated by (i) capturing a pair of images using stereocameras  (ii)  computing  disparity  using  semi-global  blockmatching  algorithm  and  (iii)  extracting  depth  infor-mation  using  these  disparity  values. Using  the  esti-mated  3D  landmarks,  facial  recognition  is  performed  us-ing Abc algorithm.  The depth maps were fine-tuned usingStructural similarity index with the help of the Stere-olab dataset, which contains stereo images with theirground truth depth information.  Unfortunately, the collec-tion of required datasets was not possible due to the currentsituation.  The approach was evaluated by checking the ac-curacy of the recognition.

## References

* Thanks to https://timosam.com/python_opencv_depthimage/, Provided the base code for creating depth map using disparity.
* Dataset : https://vision.deis.unibo.it/fede/ds-stereo-lab.html


## Getting Started
The project consists of 5 directories and 5 scripts included in it. The scripts are SelfDATA_Depth.py, StereoLab_Fine_Tuning.py, ssim.py, exc_face.py and test.py. SelfDATA_Depth is used to create the depth data for face data collected by us with all possible parameter combinations which are discussed in the paper. StereoLab_Fine_Tuning is similar to SelfDATA_Depth but this uses stereo-lab data. ssim is to check for structural similarity index values for depth maps estimated. ssim uses StereoLab_Fine_Tuning and provides the depth maps which have accuracy of above 60%. exc_face finds the face in given image and outputs the face alone as output.


### Prerequisites

Needed packages are, sklearn, numpy, cv2, imageio, mpl_toolkits, mlxtend, scipy, extract_face, matplotlib, skimage, imutils, and glob. Install all these packages for sure.

```
python -m pip install //any package name given above
```

## Running the tests

test.py is created to test the proposed face recognition technique. We need to provide a stock_data path and a input_data path in the script. The stock data path and input data path can be provided in the command line.

### Example of running test.py

Testing test.py, 

```
python test.py SELF_COLLECTED_DATA/Roh7/ SELF_COLLECTED_DATA/Sri/
```
Output is as follows

```
loading stock face data...
SELF_COLLECTED_DATA/Roh7/
computing disparity...
loading input face data...
SELF_COLLECTED_DATA/Sri/
computing disparity...
No Match. Becuase residue is: 4451.133386051766
```
Roh7 is taken as stock face in the above case and Sri as input face.

## Datasets

* Stereo-lab is used to fine tune the depth map estimation parameters
* SELF_COLLECTED_DATA is used to test the face recognition approch which was proposed by us. Collected by creating a stereo camera setup using mobile front camera.
* WILD_SELF_COLLECTED_DATA is out of this project, these are faces which are collected with uncontrolled conditions like, not a right stereo pair, uneven stereo pair, etc. 

## Authors

* **Sai Vijay Rohit Pantam** [vijayrohit](https://github.com/vijayrohit)
* **Meghana Ravirala** [meghanaravirala](https://github.com/meghanaravirala)


