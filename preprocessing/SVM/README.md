# Introduction
In On this project, we will read some news videos of new anchor talking and change it into the frame images of the lip for [this project]()

# Dlib-lip-detection
Developed by [Davis King](https://github.com/davisking), dlib is a python package for various image processing tasks.
dlib has quite useful implementations for computer vision, including:
* Facial landmark detection
* Correlation tracking
* Deep metric learning

from those, we'll use Facial landmark detection feature to detect and crop lip images from human face image.

# Installation
The easist way to get dlib and other needed libraries is using pip. open command prompt, and type:
~~~shell
$ pip install numpy
$ pip install cv2
$ pip install cmake dlib
~~~

This will automatically install the required libraries for our project and its dependencies.

# Detecting Lips on given image
## Facial landmarks on dlib
The facial landmark detector for our project produces 68 (x, y)-coordinates that map to the specific facial structures. These are trained by the [iBUG-300W dataset](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/). You can download the pretrained landmark detector model from the [detector folder](https://github.com/skaws2003/Dlib-lip-detection/tree/master/detector) on this repository

Below, we can see what the 68 facial landmarks are. 
![facial_landmarks](./captures/facial_landmarks.jpg)
from this image, we might see lip is corresponding to the landmark number [49,68].



## Detecting facial landmarks
### Preparing for detection
Now look into our example code. We first import the libraries we need:
~~~python
import dlib
import cv2
import os
~~~
dlib package will be used for facial landmark detection, of course, and cv2 will be used for image processing. os package is for reading our image file list.

Then we load face detector, facial landmark detector.
Also we will declare some variables(constants) used for the project:
~~~python
# Some constants
RESULT_PATH = './result/'       # The path that the result images will be saved
VIDEO_PATH = './dataset/'       # Dataset path
LOG_PATH = 'log.txt'            # The path for the working log file
LIP_MARGIN = 0.3                # Marginal rate for lip-only image.
RESIZE = (64,64)                # Final image size
logfile = open(LOG_PATH,'w')
# Face detector and landmark detector
face_detector = dlib.get_frontal_face_detector()   
landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")	# Landmark detector path
~~~


Since dlib produces its result in spacial format called 'shape', we need to write a function to convert it into python list:
~~~python
def shape_to_list(shape):
	coords = []
	for i in range(0, 68):
		coords.append((shape.part(i).x, shape.part(i).y))
	return coords
~~~
Now everything is ready. Let's load the video for our project.

### Facial landmark detection
We start with reading the files list on the dataset directory:
~~~python
video_list = os.listdir(VIDEO_PATH)     # Read video list
~~~

Now, we read individual video and parse it into the frames. 
The code for this task is given below:
~~~python
for vid_name in video_list:                 # Iterate on video files
    vid_path = VIDEO_PATH + vid_name
    vid = cv2.VideoCapture(vid_path)       # Read video

    # Parse into frames 
    frame_buffer = []               # A list to hold frame images
    frame_buffer_color = []         # A list to hold original frame images
    while(True):
	    success, frame = vid.read()                # Read frame
	    if not success: break                           # Break if no frame to read left
	    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)   # Convert image into grayscale
	    frame_buffer.append(gray)                  # Add image to the frame buffer
        frame_buffer_color.append(frame)
    vid.release()
~~~
The parsed frames are saved in the list `frame_buffer`.
On **line3**, we read the video by OpenCV function `cv2.VideoCapture()`.
On **line9** we read a single frame image. *success* variables indicates whether the frame reading was succesful.
If the frame reading was unsuccessful, we finish reading frames by **line10** .


Then we detect facial landmark by code below. Continued from previous code:
~~~python
    # Obtain face landmark information
    landmark_buffer = []        # A list to hold face landmark information
    for (i, image) in enumerate(frame_buffer):          # Iterate on frame buffer
        face_rects = face_detector(image,1)             # Detect face
        if len(face_rects) < 1:                 # No face detected
            print("No face detected: ",vid_path)
            logfile.write(vid_path + " : No face detected \r\n")
            break
        if len(face_rects) > 1:                  # Too many face detected
            print("Too many face: ",vid_path)
            logfile.write(vid_path + " : Too many face detected \r\n")
            break
        rect = face_rects[0]                    # Proper number of face
        landmark = landmark_detector(image, rect)   # Detect face landmarks
        landmark = shape_to_list(landmark)
        landmark_buffer.append(landmark)
~~~
The landmarks are saved on the list `landmark_buffer`, by the format(frame number, landmark list).
On __line4__, we detect face by predefined face detector(see above section _preparing for detection_).
__line5__ through __line12__ deals with the case with no face or more than two faces. On that case, we simply reject the video.
On __line14__, we find face landmarks.
On __line15__, we change the resulting format to the python list format.


Now the only remaining task is cropping. See below:
~~~python
    cropped_buffer = []
    for (i,landmark) in enumerate(landmark_buffer):
        lip_landmark = landmark[48:68]                                          # Landmark corresponding to lip
        lip_x = sorted(lip_landmark,key = lambda pointx: pointx[0])             # Lip landmark sorted for determining lip region
        lip_y = sorted(lip_landmark, key = lambda pointy: pointy[1])
        x_add = int((-lip_x[0][0]+lip_x[-1][0])*LIP_MARGIN)                     # Determine Margins for lip-only image
        y_add = int((-lip_y[0][1]+lip_y[-1][1])*LIP_MARGIN)
        crop_pos = (lip_x[0][0]-x_add, lip_x[-1][0]+x_add, lip_y[0][1]-y_add, lip_y[-1][1]+y_add)   
        cropped = frame_buffer_color[i][crop_pos[2]:crop_pos[3],crop_pos[0]:crop_pos[1]]        # Crop image
        cropped = cv2.resize(cropped,(RESIZE[0],RESIZE[1]),interpolation=cv2.INTER_CUBIC)       # Resize
	cropped_buffer.append(cropped)
~~~
The final cropped image is saved in the list `cropped_buffer`.
On __line3__, we first obtain lip landmarks from the landmark positions.
from __line4__ through __line8__, we then determine where to crop from the original frame image. The cropping task is done by sorting landmark positions by x-coordinate and y-coordinate.
from __line9__, we crop image, then is resized by cv2.resize() function.


Lastly, we save our results.
~~~python
    # Save result
    directory = RESULT_PATH + vid_name + "/"
    for (i,image) in enumerate(cropped_buffer):
        if not os.path.exists(directory):           # If the directory not exists, make it.
            os.makedirs(directory)
        cv2.imwrite(directory + "%d"%(i+1) + ".jpg", image)     # Write lip image

logfile.close()
~~~
on __line4__, we test whether there exists the path to save our results.
then on __line6__ we save our result.



