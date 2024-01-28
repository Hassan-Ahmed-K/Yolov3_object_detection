# Import Required Liberaries

import numpy as np
import cv2

# Making BoundBox Class

class BoundBox:

	def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
		self.xmin = xmin
		self.ymin = ymin
		self.xmax = xmax
		self.ymax = ymax
		self.objness = objness
		self.classes = classes
		self.label = -1
		self.score = -1
  
  
	# Function To get the index of the label having maximum score
	def get_label(self):
		if self.label == -1:
			self.label = np.argmax(self.classes)
 
		return self.label

	# Function To get the score of the given class
	def get_score(self):
		if self.score == -1:
			self.score = self.classes[self.get_label()]
 
		return self.score

# Function to apply Sigmoid Function
def _sigmoid(x):
	return 1. / (1. + np.exp(-x))

# Function to Get the bounding Box By using output of Model 
def decode_netout(netout, anchors, net_h, net_w):
	grid_h, grid_w = netout.shape[:2]
	nb_box = 3
	netout = netout.reshape((grid_h, grid_w, nb_box, -1))
	nb_class = netout.shape[-1] - 5
	boxes = []
	netout[..., :2]  = _sigmoid(netout[..., :2])
	netout[..., 4:]  = _sigmoid(netout[..., 4:])
	netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
 
	for i in range(grid_h*grid_w):
		row = i / grid_w
		col = i % grid_w
		for b in range(nb_box):
			# 4th element is objectness score
			objectness = netout[int(row)][int(col)][b][4]
			# if(objectness.all() <= obj_thresh): continue
			# first 4 elements are x, y, w, and h
			x, y, w, h = netout[int(row)][int(col)][b][:4]
			x = (col + x) / grid_w # center position, unit: image width
			y = (row + y) / grid_h # center position, unit: image height
			w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
			h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
			# last elements are class probabilities
			classes = netout[int(row)][col][b][5:]
			box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes) # Instantiate BoundBox Class
			boxes.append(box)
	return boxes

# Funcion to filter out the bounding boxes having low confidence
def box_filter(boxes,labels,threshold_socre):
	valid_boxes=[]
	valid_labels=[]
	valid_scores=[]
	for box in boxes:
		for i in range(len(labels)):
			if box.classes[i] > threshold_socre:
				valid_boxes.append(box)
				valid_labels.append(labels[i])
				valid_scores.append(box.classes[i])
	
	return (valid_boxes,valid_labels,valid_scores)


# Function to Draw the Bounding BOX On the inout Image
def draw_boxes(filename, valid_data):
    data = cv2.imread(filename)  # Read the Image
    boxes, labels = [], []
    color = (255, 255, 255)  # BGR color (white in this case)
    thickness = 2  # Thickness of the rectangle border

    for i in range(len(valid_data[0])):
        box = valid_data[0][i]
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax  # Retrieving y1, x1, y2, x2 from box

        # Making Rectangle Object
        cv2.rectangle(data, (x1, y1), (x2, y2), color, thickness)

        print(valid_data[1][i], valid_data[2][i])
        label = "%s (%.3f)" % (valid_data[1][i], valid_data[2][i])
        cv2.putText(data, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        boxes.append([x1, y1, x2, y2])
        labels.append(valid_data[1][i])
    
    
    cv2.imshow("Image", data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return boxes, labels

def draw_vedio_boxes(frame, valid_data):
  boxes,labels = [],[]
  color = (255, 255, 255)  # BGR color (green in this case)
  thickness = 2       # Thickness of the rectangle border
  for i in range(len(valid_data[0])):
    box = valid_data[0][i]
    y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax # Retriving y1, x1, y2, x2 from box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    label_text = f"{valid_data[1][i]} ({valid_data[2][i]:.3f})"
    cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    boxes.append([x1, y1, x2, y2 ])
    labels.append(valid_data[1][i])
    
  cv2.imshow("vedio",frame)
  
  return boxes ,labels

def encoder_dic(valid_data):
  data_dic={}
  (valid_boxes,valid_labels,valid_scores)=valid_data
  
  # Iterate on valid_boxes,valid_labels,valid_scores simultaneously
  for box, label,score in zip(valid_boxes,valid_labels,valid_scores):
    if label not in data_dic:
      data_dic[label]=[[score,box,'kept']]
    else:
      data_dic[label].append([score,box,'kept'])
      
  return data_dic

# Function to Return the maximum Minimum Coordinates of the bounding Box
def decode_box_coor(box):
  return (box.xmin, box.ymin,box.xmax, box.ymax )

# Function to Return the best fit Bounding Box of Detected Object
def iou(box1, box2):
   # Retriving the Bounding Box Coordinate
  (box1_x1, box1_y1, box1_x2, box1_y2) = decode_box_coor(box1)
  (box2_x1, box2_y1, box2_x2, box2_y2) = decode_box_coor(box2)

  # Finding The Coordinate of best fit bounding box
  xi1 = max(box1_x1,box2_x1)
  yi1 = max(box1_y1,box2_y1)
  xi2 = min(box1_x2,box2_x2)
  yi2 = min(box1_y2,box2_y2)
  
  # Finding width ,Height,area of Best fit Bounding Box
  inter_width = xi2-xi1
  inter_height = yi2-yi1
  inter_area = max(inter_height,0)*max(inter_width,0)
  
  box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
  box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
  union_area = box1_area+box2_area-inter_area 

  iou = inter_area/union_area
  
  return iou

# Function Non-Maximum Suppression OF Bounding Box
def do_nms(data_dic, nms_thresh):
  final_boxes,final_scores,final_labels=list(),list(),list()
  for label in data_dic:
    scores_boxes=sorted(data_dic[label],reverse=True) # Sorting Score Boxes
    for i in range(len(scores_boxes)):
      if scores_boxes[i][2]=='removed': continue
      for j in range(i+1,len(scores_boxes)):
        if iou(scores_boxes[i][1],scores_boxes[j][1]) >= nms_thresh:
          scores_boxes[j][2]="removed"

    for e in scores_boxes:
      print(label+' '+str(e[0]) + " status: "+ e[2])
      
      # Appending Final Result and return in the For, of Tuple
      if e[2]=='kept':
        final_boxes.append(e[1])
        final_labels.append(label)
        final_scores.append(e[0])
    

  return (final_boxes,final_labels,final_scores)

