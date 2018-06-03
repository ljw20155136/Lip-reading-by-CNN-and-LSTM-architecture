import dlib
import cv2
import os

# Some constants
RESULT_PATH = './result/'       # The path that the result images will be saved
VIDEO_PATH = './dataset/'       # Dataset path
LOG_PATH = 'log.txt'            # The path for the working log file
LIP_MARGIN = 0.3                # Marginal rate for lip-only image.
RESIZE = (64,64)                # Final image size
logfile = open(LOG_PATH,'w')
# Face detector and landmark detector
face_detector = dlib.get_frontal_face_detector()   
landmark_detector = dlib.shape_predictor("detector/shape_predictor_68_face_landmarks.dat")


def shape_to_list(shape):
	coords = []
	for i in range(0, 68):
		coords.append((shape.part(i).x, shape.part(i).y))
	return coords

video_list = os.listdir(VIDEO_PATH)     # Read video list

for vid_name in video_list:                 # Iterate on video files
    vid_path = VIDEO_PATH + vid_name
    vid = cv2.VideoCapture(vid_path)       # Read video

    # Parse into frames 
    frame_buffer = []               # A list to hold frame images
    frame_buffer_color = []         # A list to hold original frame images
    while(True):
        success, frame = vid.read()                # Read frame
        if not success:
            break                           # Break if no frame to read left
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)   # Convert image into grayscale
        frame_buffer.append(gray)                  # Add image to the frame buffer
        frame_buffer_color.append(frame)
    vid.release()

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

    # Crop images
    cropped_buffer = []
    for (i,landmark) in enumerate(landmark_buffer):
        lip_landmark = landmark[48:68]                                          # Landmark corresponding to lip
        lip_x = sorted(lip_landmark,key = lambda pointx: pointx[0])             # Lip landmark sorted for determining lip region
        lip_y = sorted(lip_landmark, key = lambda pointy: pointy[1])
        x_add = int((-lip_x[0][0]+lip_x[-1][0])*LIP_MARGIN)                     # Determine Margins for lip-only image
        y_add = int((-lip_y[0][1]+lip_y[-1][1])*LIP_MARGIN)
        crop_pos = (lip_x[0][0]-x_add, lip_x[-1][0]+x_add, lip_y[0][1]-y_add, lip_y[-1][1]+y_add)   # Crop image
        cropped = frame_buffer_color[i][crop_pos[2]:crop_pos[3],crop_pos[0]:crop_pos[1]]
        cropped = cv2.resize(cropped,(RESIZE[0],RESIZE[1]),interpolation=cv2.INTER_CUBIC)        # Resize
        cropped_buffer.append(cropped)

    # Save result
    directory = RESULT_PATH + vid_name + "/"
    for (i,image) in enumerate(cropped_buffer):
        if not os.path.exists(directory):           # If the directory not exists, make it.
            os.makedirs(directory)
        cv2.imwrite(directory + "%d"%(i+1) + ".jpg", image)     # Write lip image

logfile.close()

    
