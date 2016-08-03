""" capture_faces.py
Author: Gonçalo S. Martins

This script implements the basic face detection and extraction features
needed for this work. The main section of the script illustrates its usage
in extracting faces from pre-recorded videos.

This script was tested using Python 3.5 and OpenCV 3.1.

Video capture code inspired by:
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html

Face extraction code inspired by:
http://docs.opencv.org/master/d7/d8b/tutorial_py_face_detection.html#gsc.tab=0

"Haar inversion method" inspired by work of the CASIR project:
http://mrl.isr.uc.pt/projects/casir/
"""
# Standard Library imports
import time
import os

# Non-standard imports
import cv2
import numpy


# Data loading functions
def capture_video(duration):
    """ This function captures video from the webcam for the given duration, in
    seconds. It returns the individual frames that it has captured.
    """
    # Open capture device
    cap = cv2.VideoCapture(0)
    print("Camera is operating at {} fps.".format(cap.get(cv2.CAP_PROP_FPS)))

    # Test capture device
    ret, frame = cap.read()
    if not ret:
        print("Could not capture image!")
        return

    # Start capture
    start = time.time()
    frames = []
    while(time.time() - start < duration):
        # Capture frame
        ret, frame = cap.read()

        # Append to our frame list
        frames.append(frame)

    # When everything done, release the capture
    cap.release()

    # Print frequencies
    print("True average frequency of captured frames: {}fps.".format(len(frames)/duration))

    # Return the captured frames
    return frames

def load_frames_from_video(filename):
    # Open capture device
    cap = cv2.VideoCapture(filename)

    # Extract frames and append to frame list
    frames = []
    while True:
    #for i in range(50):
        ret, frame = cap.read()
        if ret == False:
            break
        frames.append(frame)

    # Return list
    return frames


# Batch operations
def show_frames(frames, freq = 12):
    """ This function receives a list of frames and plays them back at the given
    frequency.
    """
    for frame in frames:
        cv2.imshow('frame',frame)
        cv2.waitKey(round(1000/freq))


def find_faces(frames):
    """ This function receives a list of frames and draws rectangles around all
    of the faces it finds. For now, it edits the original frames. """
    # Load the haar cascades
    front_cascade = cv2.CascadeClassifier('/home/vsantos/opencv_install/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')
    side_cascade = cv2.CascadeClassifier('/home/vsantos/opencv_install/share/OpenCV/haarcascades/haarcascade_profileface.xml')

    # Define scale factor.
    # We will reduce images by this factor to speed up processing.
    scale_factor = 0.5

    # Detect faces in all the frames
    i = 1
    face_boxes = []
    for frame in frames:
        # Print the current frame index
        print("Processing frame {} of {}.".format(i, len(frames)))
        i+=1
        # Convert image to grayscale and reduce it
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (0,0), fx=scale_factor, fy=scale_factor)

        # Detect faces
        face_list = []
        face_list.append(front_cascade.detectMultiScale(small))
        face_list.append(side_cascade.detectMultiScale(small))
        flipped_faces = side_cascade.detectMultiScale(cv2.flip(small,1))
        # Since the coordinates are flipped as well, we need to fix them:
        face_list.append([(small.shape[1] - x - w, y, w, h) for (x, y, w, h) in flipped_faces])

        # Transform faces to the original coordinate frame and append to the
        # general face list
        faces = []
        for l in face_list:
            l = [[int(x*(1/scale_factor)) for x in v] for v in l]
            faces += l

        # Detect and remove overlapping rectagles (keeping the largest)
        # TODO: Actually implement
        if type(faces) != type(tuple()):
            faces = faces[numpy.random.randint(len(faces))]
        face_boxes.append(faces)
        # Paint rectangles in images
        #for (x,y,w,h) in faces:
        #(x,y,w,h) = faces
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    return face_boxes


def crop_faces(frames, boxes):
    """ This function receives a list of frames and a list of bounding boxes
    corresponding to faces, and uses that information to produce a list of
    cropped faces.
    """
    cropped = []
    for i, frame in enumerate(frames):
        (x,y,w,h) = boxes[i]
        cropped.append(frame[y: y + h, x: x + w])

    return cropped


# Real Time
def find_faces_rt(duration=30):
    """ This function reads frames from the camera and marks faces, doing so
    continuously for a set duration. Its goal is to help set all of the
    relevant parameters.
    """
    # Open capture device
    cap = cv2.VideoCapture(0)

    # Load cascade
    front_cascade = cv2.CascadeClassifier('/home/vsantos/opencv_install/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')
    side_cascade = cv2.CascadeClassifier('/home/vsantos/opencv_install/share/OpenCV/haarcascades/haarcascade_profileface.xml')

    # Define scale factor.
    # We will reduce images by this factor to speed up processing.
    scale_factor = 0.5

    # Test capture device
    ret, frame = cap.read()
    if not ret:
        print("Could not capture image!")
        return

    # Start capture
    start = time.time()
    frames = []
    while(time.time() - start < duration):
        # Capture frame
        ret, frame = cap.read()

        # Convert image to grayscale and reduce it
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (0,0), fx=scale_factor, fy=scale_factor)

        # Detect faces
        face_list = []
        face_list.append(front_cascade.detectMultiScale(small))
        face_list.append(side_cascade.detectMultiScale(small))
        flipped_faces = side_cascade.detectMultiScale(cv2.flip(small,1))
        # Since the coordinates are flipped as well, we need to fix them:
        face_list.append([(small.shape[1] - x - w, y, w, h) for (x, y, w, h) in flipped_faces])

        # Transform faces to the original coordinate frame and append to the
        # general face list
        faces = []
        for l in face_list:
            l = [[int(x*(1/scale_factor)) for x in v] for v in l]
            faces += l

        # Detect and remove overlapping rectagles (keeping the largest)

        # Paint rectangles in images
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imshow('frame',frame)
        cv2.waitKey(round(10))

    # When everything done, release the capture
    cap.release()


# Pipeline
def find_face(frame):
    """ This function finds a face in a given image. """
    # Load the haar cascades
    front_cascade = cv2.CascadeClassifier('/home/vsantos/opencv_install/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')
    side_cascade = cv2.CascadeClassifier('/home/vsantos/opencv_install/share/OpenCV/haarcascades/haarcascade_profileface.xml')

    # Define scale factor.
    # We will reduce images by this factor to speed up processing.
    scale_factor = 0.25

    # Detect faces in all the frames
    # Convert image to grayscale and reduce it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (0,0), fx=scale_factor, fy=scale_factor)

    # Detect faces
    face_list = []
    face_list.append(front_cascade.detectMultiScale(small))
    face_list.append(side_cascade.detectMultiScale(small))
    flipped_faces = side_cascade.detectMultiScale(cv2.flip(small,1))
    # Since the coordinates are flipped as well, we need to fix them:
    face_list.append([(small.shape[1] - x - w, y, w, h) for (x, y, w, h) in flipped_faces])

    # Transform faces to the original coordinate frame and append to the
    # general face list
    faces = []
    for l in face_list:
        l = [[int(x*(1/scale_factor)) for x in v] for v in l]
        faces += l

    # Detect and remove overlapping rectagles (keeping the largest)
    # TODO: Actually implement
    if type(faces) != type(tuple()) and len(faces) > 0:
        faces = faces[numpy.random.randint(len(faces))]

    # Return the face we found
    return faces


def crop_face(frame, box):
    """ This function takes an image and a bounding box and crops the image. """
    (x,y,w,h) = box
    return frame[y: y + h, x: x + w]


def extract_faces_from_video(filename, save_location):
    """ This function receives a video file and extracts all of the faces in it,
    saving the images into the provided folder.
    """
    # Ensure that the save location exists
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    # Open capture device
    cap = cv2.VideoCapture(filename)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Process frame-by-frame
    i = 1
    while True:
        # Check clock
        start = time.time()

        # Get frame
        ret, frame = cap.read()
        if ret == False:
            print("Could not read frame, stopping!")
            break

        # Find face
        box = find_face(frame)

        if len(box) > 0:
            # Crop face
            cropped = crop_face(frame, box)

            # Save to file
            cv2.imwrite(os.path.join(save_location, "{}.jpg".format(i)), cropped)

        # Inform on time
        fps = 1/(time.time()-start)
        eta = (num_frames-i)/fps
        print("Processed frame {:4d} at {:1.2f} fps. ETA: {:1.2f}s".format(i, fps, eta))

        # Increment counter
        i += 1


if __name__ == "__main__":
    #find_faces_rt()
    #frames = capture_video(5)
    #frames = load_frames_from_video("2016-07-12-131940.webm")
    #frames = load_frames_from_video("20160712_134926.mp4")
    #boxes = find_faces(frames)
    #cropped = crop_faces(frames, boxes)
    #show_frames(frames)
    #show_frames(cropped)

    #extract_faces_from_video("videos_vvb/Filipa_HNS.mp4", "./Filipa_HNS")
    #extract_faces_from_video("videos_vvb/Filipa_HNY.mp4", "./Filipa_HNY")
    #extract_faces_from_video("videos_vvb/Filipa_HNN.mp4", "./Filipa_HNN")
    #extract_faces_from_video("videos_vvb/Filipa_HSS.mp4", "./Filipa_HSS")
    #extract_faces_from_video("videos_vvb/Filipa_HSY.mp4", "./Filipa_HSY")
    #extract_faces_from_video("videos_vvb/Filipa_HSN.mp4", "./Filipa_HSN")
    #extract_faces_from_video("videos_vvb/Gonçalo_HNS.mp4", "./Gonçalo_HNS")
    #extract_faces_from_video("videos_vvb/Gonçalo_HNY.mp4", "./Gonçalo_HNY")
    #extract_faces_from_video("videos_vvb/Gonçalo_HNN.mp4", "./Gonçalo_HNN")
    #extract_faces_from_video("videos_vvb/Gonçalo_HSS.mp4", "./Gonçalo_HSS")
    #extract_faces_from_video("videos_vvb/Gonçalo_HSY.mp4", "./Gonçalo_HSY")
    #extract_faces_from_video("videos_vvb/Gonçalo_HSN.mp4", "./Gonçalo_HSN")
    #extract_faces_from_video("videos_vvb/Gonçalo_test.mp4", "./Gonçalo_test")
    #extract_faces_from_video("videos_vvb/Filipa_test.mp4", "./Filipa_test")
