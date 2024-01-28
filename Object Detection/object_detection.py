#Importing Required Libraries

from numpy import expand_dims
from PIL import Image
from src.model import build_yolov3_model,WeightReader
from src.image_processing import *
from src.bound_box import *
from src.util import *
import cv2



# Declearing Labels

labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
	"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
	"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
	"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
	"apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
	"chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# Declearing Bounding Box Anchor
anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]


# Predict Function which take image as Input and Detect object by using YOLOV3
def predict(image_path,IMAGE_WIDTH=416,IMAGE_HEIGHT=416,labels = labels,anchors = anchors):
    try:
      if(("https" in image_path) or ("http" in image_path)): # Condition To check If URL is given as input
          destination_path = f'/home/hassan-ahmed-khan/Ai/Yolo/Object_Detection/images/download/{image_path.split("/")[-1]}'
          download_image(image_path,destination_path)
          image_path = destination_path
      image,image_w,image_h=load_and_preprocess_image(image_path,[IMAGE_WIDTH,IMAGE_HEIGHT]) # Applying load_and_preprocess_image function
      image = expand_dims(image, 0) # Flatten Image Array
      yhat = model.predict(image) # Detecting object by using tensorflow predict function
      boxes = list()
      for i in range(len(yhat)):
          boxes += decode_netout(yhat[i][0], anchors[i], net_h=IMAGE_HEIGHT, net_w=IMAGE_WIDTH) # Applying decode_netout function
      for i in range(len(boxes)):
          x_offset, x_scale = (IMAGE_WIDTH - IMAGE_WIDTH)/2./IMAGE_HEIGHT, float(IMAGE_WIDTH)/IMAGE_WIDTH
          y_offset, y_scale = (IMAGE_HEIGHT - IMAGE_HEIGHT)/2./IMAGE_HEIGHT, float(IMAGE_HEIGHT)/IMAGE_HEIGHT
          boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
          boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
          boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
          boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)
          
      valid_data= box_filter(boxes, labels, threshold_socre=0.6) # Applying box_filter function
      dic=encoder_dic(valid_data) # Applying encoder_dic function
      final_data=do_nms(dic, 0.7) # Applying do_nms function
      boxes,labels = draw_boxes(image_path,final_data) # Applying draw_boxes function
    except:
        print("Error Occur")
        
        
def test(image_path = "/home/hassan-ahmed-khan/Ai/Yolo/Object_Detection/images/zebra_horse.jpg",save=False,IMAGE_WIDTH=416,IMAGE_HEIGHT=416,labels = labels,anchors = anchors):
  try:
     if(("https" in image_path) or ("http" in image_path)):
          destination_path = f'/home/hassan-ahmed-khan/Ai/Yolo/Object_Detection/images/download/{image_path.split("/")[-1]}'
          download_image(image_path,destination_path)
          image_path = destination_path
          
     image,image_w,image_h=load_and_preprocess_image(image_path,[IMAGE_WIDTH,IMAGE_HEIGHT]) # Applying load_and_preprocess_image function
     image = expand_dims(image, 0)
     yhat = model.predict(image)
     boxes = list()
     for i in range(len(yhat)):
         boxes += decode_netout(yhat[i][0], anchors[i], net_h=IMAGE_HEIGHT, net_w=IMAGE_WIDTH) # Applying decode_netout function
     for i in range(len(boxes)):
         x_offset, x_scale = (IMAGE_WIDTH - IMAGE_WIDTH)/2./IMAGE_HEIGHT, float(IMAGE_WIDTH)/IMAGE_WIDTH
         y_offset, y_scale = (IMAGE_HEIGHT - IMAGE_HEIGHT)/2./IMAGE_HEIGHT, float(IMAGE_HEIGHT)/IMAGE_HEIGHT
         boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
         boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
         boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
         boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)
    
     valid_data= box_filter(boxes, labels, threshold_socre=0.6) # Applying box_filter function
     dic=encoder_dic(valid_data) # Applying encoder_dic function
     final_data=do_nms(dic, 0.7) # Applying do_nms function
     boxes,labels = draw_boxes(image_path,final_data) # Applying draw_boxes function
     cropped_detected_object(image_path,boxes,labels,save=True) # Applying cropped_detected_object function
 
  except:
      print("Error Occur")
  
    
def vedio_object_detected(frame,IMAGE_WIDTH=416,IMAGE_HEIGHT=416,labels = labels,anchors = anchors):
    try:
      
      image,image_w,image_h=load_and_preprocess_frame(frame,[IMAGE_WIDTH,IMAGE_HEIGHT]) # Applying load_and_preprocess_image function
      image = expand_dims(image, 0) # Flatten Image Array
      yhat = model.predict(image) # Detecting object by using tensorflow predict function
      boxes = list()
      for i in range(len(yhat)):
          boxes += decode_netout(yhat[i][0], anchors[i], net_h=IMAGE_HEIGHT, net_w=IMAGE_WIDTH) # Applying decode_netout function
      for i in range(len(boxes)):
          x_offset, x_scale = (IMAGE_WIDTH - IMAGE_WIDTH)/2./IMAGE_HEIGHT, float(IMAGE_WIDTH)/IMAGE_WIDTH
          y_offset, y_scale = (IMAGE_HEIGHT - IMAGE_HEIGHT)/2./IMAGE_HEIGHT, float(IMAGE_HEIGHT)/IMAGE_HEIGHT
          boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
          boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
          boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
          boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)
      
      valid_data= box_filter(boxes, labels, threshold_socre=0.6) # Applying box_filter function
      dic=encoder_dic(valid_data) # Applying encoder_dic function
      final_data=do_nms(dic, 0.7) # Applying do_nms function
      boxes,labels = draw_vedio_boxes(frame,final_data) # Applying draw_boxes function
    except Exception as e:
        print(f"Error Occurred: {e}")


def predict_vedio(video_path):
    # Assuming 'cap' is your video capture object
    video_cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = video_cap.read()
        if not ret:
            break  
        vedio_object_detected(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    
    # photo_filename='images/zebra_horse.jpg'
    # image_url = 'https://storagecdn.strathcona.ca/files/filer_public_thumbnails/images/tas-medium-trafficsignals-intersection-660x396.jpg__660.0x396.0_q85_subsampling-2.jpg'
    
    model = build_yolov3_model() # Making Yolo3 Model
    weight_reader = WeightReader('/home/hassan-ahmed-khan/Ai/Yolo/Object_Detection/model/yolov3.weights') # Lading the weights on the above model
    weight_reader.load_weights(model)
    
    video_path = '/home/hassan-ahmed-khan/Ai/Yolo/Object_Detection/images/video.mp4'
    print("Leave Empty If Want Default image for test")
    photo_filename = input("Enter Image Path: ")  # Take Image URI or URL as Input
    
    if (photo_filename == ""):
        test() # Applying Test Function
    
    else:
        
        # predict_vedio(video_path)
        # test(photo_filename) # Applying Test Function
        predict(photo_filename) # Applying Predict Function
        

# 

# # print(model.summary())

# model.save("model/yolov3.h5")



