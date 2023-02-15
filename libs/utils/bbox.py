################################################################
# Dependencies: System
################################################################
import numpy as np
from PIL import Image

################################################################
# Dependencies: Local
################################################################
from utils import ios

################################################################
# Functions: Bounding boxes
################################################################

def convert_from_results(xmin, ymin, width, height):
  xmax = xmin + width
  ymax = ymin + height
  return np.array([xmin,ymin,xmax,ymax])



def convert_from_norm(fn_img, fn_lbl):
  #from: xcenter,ycenter,width,height
  #to: xmin,ymin,xmax,ymax
  img = Image.open(fn_img)
  W, H = img.size
  content = ios.read_txt_to_list(fn_lbl)
  classes = []
  bboxes = None

  for bbox in content:
    label,xcenter,ycenter,width,height = bbox.split(" ")
    classes.append(int(label))
    
    xmin = (float(xcenter) - float(width)/2.)*W
    xmax = (float(xcenter) + float(width)/2.)*W
    ymin = (float(ycenter) - float(height)/2.)*H
    ymax = (float(ycenter) + float(height)/2.)*H

    tmp = np.array([xmin,ymin,xmax,ymax])
    if bboxes is None:
      bboxes = tmp.reshape(1,4)
    else:
      bboxes = np.vstack((bboxes,tmp))

  return classes,bboxes
