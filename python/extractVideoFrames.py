# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:25:42 2016

@author: jimmy

play video
"""

import numpy as np
import cv2
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--video_file', type = str, required = True)
parser.add_argument('--start_frame', type = int, required = True)
parser.add_argument('--end_frame', type = int, required = True)
parser.add_argument('--set_fps', type = int, required = True)
parser.add_argument('--step', type = int, required = True)
parser.add_argument('--width', type = int, required = False, default = 320)
parser.add_argument('--height', type = int, required = False, default = 180)
parser.add_argument('--save_dir', type = str, required = True)
args = parser.parse_args()

video_file_name = args.video_file
start_frame = args.start_frame
end_frame = args.end_frame
fps_set = args.set_fps
step = args.step
width = args.width
height = args.height
save_dir = args.save_dir


cap = cv2.VideoCapture(video_file_name)

if cap.isOpened() != True:
    print('Error, can not read from %s' % video_file_name)
    exit()
    
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps    = cap.get(cv2.CAP_PROP_FPS)

print('FPS is %f\n' % fps)
print('set FPS is %f\n' % fps_set)
print('width, height is %d %d\n' % (width, height))

def millisecondsFromIndex(index, fps):
    return index * 1000.0/fps;

def getFrameByIndex(cap, index, fps = 60.0):
    t = millisecondsFromIndex(index, fps)
    cap.set(cv2.CAP_PROP_POS_MSEC, t)
    ret, frame = cap.read()
    if ret == True:
        return frame
    else:
        return None

# read index from files
indices = []
for i in range(start_frame, end_frame, step):
    indices.append(i)    
    
print('frame number is %d' % len(indices))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for i in indices:
    frame = getFrameByIndex(cap, i, fps_set)     
    if frame != None:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
        name = save_dir + '/%08d.jpg' % i
        cv2.imwrite(name, frame)
        print('save to %s' % name)
        
cap.release()
    
    
