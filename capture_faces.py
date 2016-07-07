""" This script opens the webcam and captures video for a set length of time.
Then, it detects and crops faces on all of the captured frames, and saves the
image files to a folder.

Video capture code inspired by:
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html

Face extraction code inspired by:
http://docs.opencv.org/master/d7/d8b/tutorial_py_face_detection.html#gsc.tab=0
"""
import cv2
import time

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
    front_cascade = cv2.CascadeClassifier('/home/vsantos/opencv_install/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    #side_cascade = cv2.CascadeClassifier('/home/vsantos/opencv_install/share/OpenCV/haarcascades/haarcascade_profileface.xml')

    # Detect faces in all the frames
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = front_cascade.detectMultiScale(gray)#, 1.3, 5)
        side_faces = side_cascade.detectMultiScale(gray)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        #for (x,y,w,h) in side_faces:
        #    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

def find_faces_rt(duration=30):
    """ This function reads frames from the camera and marks faces, doing so
    continuously for a set duration. Its goal is to help set all of the
    relevant parameters.
    """
    # Open capture device
    cap = cv2.VideoCapture(0)

    # Load cascade
    front_cascade = cv2.CascadeClassifier('/home/vsantos/opencv_install/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = front_cascade.detectMultiScale(gray)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imshow('frame',frame)
        cv2.waitKey(round(10))

    # When everything done, release the capture
    cap.release()


if __name__ == "__main__":
    #find_faces_rt()
    frames = capture_video(5)
    find_faces(frames)
    show_frames(frames)
