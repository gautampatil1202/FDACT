import cv2

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import imutils
from playsound import playsound

from tkinter import *
import tkinter as tk
import cv2 
from PIL import Image, ImageTk 
import paho.mqtt.client as mqtt



MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


Alarm = False



def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  personCount = 0
  for detection in detection_result.detections:
    # Draw bounding_box
    if(detection.categories[0].category_name == 'person'):
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        if(category_name=='person'):
            personCount = personCount+1
        text_location = (MARGIN + bbox.origin_x,
                         MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    
  cv2.putText(image, "Person:" + str(personCount), (10, 40), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 255, 255), 2)
  return image, personCount;



def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe("reco",1)


def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))

broker_address="broker.hivemq.com" 
port=1883
#broker_address="iot.eclipse.org" #use external broker
client = mqtt.Client("356sflsdsdfsdgdfuhafoasp123") #create new instance

client.on_connect = on_connect
client.on_message = on_message
client.connect(broker_address,port) #connect to broker

fire_cascade = cv2.CascadeClassifier(r'fire_detection_cascade_model.xml') # To access xml file which includes positive and negative images of fire. (Trained images)	
def DetecFire(frame):
    Alarm_Status = False
   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # To convert frame into gray color
    fire = fire_cascade.detectMultiScale(frame, 1.2, 5) # to provide frame resolution
    
    ## to highlight fire with square 
    for (x,y,w,h) in fire:
        #cv2.rectangle(frame,(x-20,y-20),(x+w+20,y+h+20),(255,0,0),2)
        #roi_gray = gray[y:y+h, x:x+w]
        #roi_color = frame[y:y+h, x:x+w]
         # Send a signal to Arduino to blink the light
        if len(fire) > 0.5:
            # Code to send a signal to Arduino to blink the light
            #print("Fire alarm initiated")
            #print("Bazar alarm initiated")
            return True
            
            #ser.write(b'1')
        

    #cv2.imshow('frame', frame)
    return False
    


# The function cv2.imshow() is used to display an image in a window.

# STEP 2: Create an ObjectDetector object.
base_options = python.BaseOptions(model_asset_path='efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)
print('detector')







# Create a GUI app 
app = Tk() 
app.geometry("800x600")
# Bind the app with Escape keyboard to 
# quit app whenever pressed 
app.bind('<Escape>', lambda e: app.quit()) 

# Create a label and display it on app 
label_widget = Label(app) 
label_widget.pack() 

Msgtext = tk.StringVar() 
Persontext = tk.StringVar() 
Alarmtext = tk.StringVar() 

l2 = Label(app, textvariable = Msgtext)
l2.pack();
l3 = Label(app, textvariable = Persontext)
l3.pack();
l4 = Label(app, textvariable = Alarmtext)
l4.pack();
Msgtext.set("") 
vid = cv2.VideoCapture(0) 

def open_camera(): 
    global Alarm
    global Msgtext
    global Persontext
    global Alarmtext
    # Capture the video frame by frame 
    _, frame = vid.read() 
    Msgtext.set("Runnig") 
  
    
    #ret, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    FireStatus = DetecFire(frame)
    if(FireStatus == True and Alarm == False):
        Alarm = True
        playsound(r'Fire_alarm.mp3',False)
    
    print("FireStatus" + str(FireStatus))
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    # STEP 4: Detect objects in the input image.
    detection_result = detector.detect(image)
    #print("persons = " + str(persons))
    # STEP 5: Process the detection result. In this case, visualize it.
    image_copy = np.copy(image.numpy_view())
    annotated_image, persons = visualize(image_copy, detection_result)
    print(persons)
    Persontext.set("Total Person: " + str(persons))
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    client.loop_start()
    client.publish("india/pune/personcount","Total Person: " + str(persons))
    if(Alarm == True):
        client.publish("india/pune/firestatus","Fire Detected")
        Persontext.set("Fire Alert: fire detected please evacuate")
        cv2.putText(annotated_image, "Fire Detected", (10, 80), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 0, 266), 3)
    else:
        client.publish("india/pune/firestatus","No Fire")
        
    #cv2.imshow('Process', annotated_image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        pass
  # Convert image from one color space to other 
    opencv_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGBA) 

    # Capture the latest frame and transform to image 
    captured_image = Image.fromarray(opencv_image) 

    # Convert captured image to photoimage 
    photo_image = ImageTk.PhotoImage(image=captured_image)
    print("loop")
    label_widget.photo_image = photo_image 

    # Configure image in the label 
    label_widget.configure(image=photo_image) 

    # Repeat the same process after every 10 seconds 
    label_widget.after(10, open_camera) 

# Create a button to open the camera in GUI app 
button1 = Button(app, text="Open Camera", command=open_camera) 
button1.pack() 

# Create an infinite loop for displaying app on screen 
app.mainloop() 




# cap = cv2.VideoCapture(0)
# while True:
    # ret, frame = cap.read()
    # frame = imutils.resize(frame, width=600)
    # FireStatus = DetecFire(frame)
    # if(FireStatus == True and Alarm == False):
        # Alarm = True
        # playsound(r'Fire_alarm.mp3',False)
    
    # print("FireStatus" + str(FireStatus))
    # image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    # detection_result = detector.detect(image)
    #print("persons = " + str(persons))
    # image_copy = np.copy(image.numpy_view())
    # annotated_image, persons = visualize(image_copy, detection_result)
    # print(persons)
    # rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    # if(Alarm == True):
        # cv2.putText(annotated_image, "Fire Detected", (10, 80), cv2.FONT_HERSHEY_PLAIN,
                    # 2, (0, 0, 266), 3)
    # cv2.imshow('Process', annotated_image)
    # key = cv2.waitKey(1)
    # if key == ord('q'):
        # break
