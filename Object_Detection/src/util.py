#Importing Required Liberaries
import requests
import os
import wget

#Function TO Downloade Image from URL
def download_image(url, destination_path):
    response = requests.get(url) # Making Access request
    if response.status_code == 200:
        with open(destination_path, 'wb') as file:
            file.write(response.content) # Downlading The Image
        print("Image downloaded successfully")
    else:
        print(f"Failed to download image. Status code: {response.status_code}")
        
        

def download_weight(URL="https://pjreddie.com/media/files/yolov3.weights", destination_path="/home/hassan-ahmed-khan/Ai/Yolo/Object_Detection/model/yolov3.weights"):
    if "yolov3.weights" not in os.listdir("/home/hassan-ahmed-khan/Ai/Yolo/Object_Detection/model"):
        try:
            print("Downloading ...")
            wget.download(URL, destination_path)
            print(f"Weights downloaded successfully to {destination_path}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Weights already exist.")




