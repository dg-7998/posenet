import tensorflow as tf
import cv2
import time
import argparse
import math
import posenet
import numpy as np
from functools import partial
import threading


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str,default='E:/Downloads/output_0.mp4',help="Optionally use a video file instead of a live camera")
args = parser.parse_args()
PART_NAMES = {
    "nose":0, "leftEye":1, "rightEye":2, "leftEar":3, "rightEar":4, "leftShoulder":5,
    "rightShoulder":6, "leftElbow":7, "rightElbow":8, "leftWrist":9, "rightWrist":10,
    "leftHip":11, "rightHip":12, "leftKnee":13, "rightKnee":14, "leftAnkle":15, "rightAnkle":16
}


def lowpass(input,moving_arr,index,size):
    if(index<size-1):
        index+=1
        moving_arr[index] = input
    elif(index>=size-1):
        index+=1
        index = index%10
        moving_arr[index] = input
    return np.sum(moving_arr)/size

STATES = {"pause":0, "up":1, "down":2}
present_state = {"left_arm":0, "right_arm":1}
MAX = {"left_angle":0, "right_angle":1, "left_speed":2, "right_speed":3}
MIN = {"left_angle":0, "right_angle":1, "left_speed":2, "right_speed":3}
NORM = {"left_angle":0, "right_angle":1, "left_speed":2, "right_speed":3}
DATA = {"left_angle":0, "right_angle":1, "left_speed":2, "right_speed":3, "avg_left_speed":4, "avg_right_speed":5}
temp1 =0
temp2 =0

def pause(x):
    '''
    if(DATA(["left_angle"]>10 or DATA["left_speed"]>0.5):
        present_state["left_arm"] = STATES["down"]
    elif(DATA["left_angle"]< 115 or DATA["left_speed"]<-0.1):
        present_state["left_arm"] = STATES["up"]
    '''
    if(x==0):
        if(NORM["left_angle"]>0.55 and DATA["left_speed"]>5):
            present_state["left_arm"] = STATES["down"]
        elif(NORM["left_angle"]<0.90 and DATA["left_speed"]<-5):
            present_state["left_arm"] = STATES["up"]
        return present_state["left_arm"]
    elif(x==1):
        if(NORM["right_angle"]>0.55 and DATA["right_speed"]>5):
            present_state["right_arm"] = STATES["down"]
        elif(NORM["right_angle"]<0.90 and DATA["right_speed"]<-5):
            present_state["right_arm"] = STATES["up"]
        return present_state["right_arm"]

def up(x):
    
    if(DATA["left_angle"]<15):
        present_state["left_arm"] = STATES["pause"]
    if(DATA["left_speed"]<-4):
        print("Move your hand slowly")
    if(DATA["left_speed"]>1):
        present_state["left_arm"] = STATES["down"]
        print("Pause State missed...")
    
    if(x==0):
        if(NORM["left_angle"]<0.55 ):
            present_state["left_arm"] = STATES["pause"]
        elif(NORM["left_angle"]>0.55 and DATA["left_speed"]>5):
            present_state["left_arm"] = STATES["down"]
            print("pause state missed")
        return present_state["left_arm"]
    elif(x==1):
        if(NORM["right_angle"]<0.55 ):
            present_state["right_arm"] = STATES["pause"]
        elif(NORM["right_angle"]>0.55 and DATA["right_speed"]>5):
            present_state["right_arm"] = STATES["down"]
            print("pause state missed")
        return present_state["right_arm"]

def down(x):
    if(x==0):
        if(NORM["left_angle"]>0.90):
            present_state["left_arm"] = STATES["pause"]
        elif(NORM["left_angle"]<0.90 and DATA["left_speed"]<-5):
            present_state["left_arm"] = STATES["up"]
            print("Pause state missed")
        return present_state["left_arm"]
    elif(x==1):
        if(NORM["right_angle"]>0.90):
            present_state["right_arm"] = STATES["pause"]
        elif(NORM["right_angle"]<0.90 and DATA["right_speed"]<-5):
            present_state["right_arm"] = STATES["up"]
            print("Pause state missed")
        return present_state["right_arm"]

def fsm_arm(x,arm):
    switcher = {0:partial(pause,arm), 1:partial(up,arm), 2:partial(down,arm)}
    func=switcher.get(x)
    return func()

def init_param():
    MIN["left_angle"] = 1000
    MIN["left_speed"] = 1000
    MIN["right_angle"] = 1000
    MIN["right_speed"] = 1000

    MAX["left_angle"] = -1000
    MAX["left_speed"] = -1000
    MAX["right_angle"] = -1000
    MAX["right_speed"] = -1000
    ti =0
    tf =0
    #temp1 =0
    #temp2 =0

def update_max():
    if(MAX["left_angle"] <= DATA["left_angle"]):
        MAX["left_angle"] = DATA["left_angle"]

    if(MAX["right_angle"] <= DATA["right_angle"]):
        MAX["right_angle"] = DATA["right_angle"]

    if(MAX["left_speed"] <= DATA["left_speed"]):
        MAX["left_speed"] = DATA["left_speed"]
    
    if(MAX["right_speed"] <= DATA["right_speed"]):
        MAX["right_speed"] = DATA["right_speed"]

def update_min():
    if(MIN["left_angle"] >= DATA["left_angle"]):
        MIN["left_angle"] = DATA["left_angle"]

    if(MIN["right_angle"] >= DATA["right_angle"]):
        MIN["right_angle"] = DATA["right_angle"]

    if(MIN["left_speed"] >= DATA["left_speed"]):
        MIN["left_speed"] = DATA["left_speed"]
    
    if(MIN["right_speed"] >= DATA["right_speed"]):
        MIN["right_speed"] = DATA["right_speed"]
    
def get_angle(arr,num):
    dist1 = 0
    dist2 = 0
    dist3 = 0
    if(num==0):
        x1 = arr[0,PART_NAMES["leftWrist"],0]
        y1 = arr[0,PART_NAMES["leftWrist"],1]
        x2 = arr[0,PART_NAMES["leftElbow"],0]
        y2 = arr[0,PART_NAMES["leftElbow"],1]
        x3 = arr[0,PART_NAMES["leftShoulder"],0]
        y3 = arr[0,PART_NAMES["leftShoulder"],1]
    elif(num==1):
        x1 = arr[0,PART_NAMES["rightWrist"],0]
        y1 = arr[0,PART_NAMES["rightWrist"],1]
        x2 = arr[0,PART_NAMES["rightElbow"],0]
        y2 = arr[0,PART_NAMES["rightElbow"],1]
        x3 = arr[0,PART_NAMES["rightShoulder"],0]
        y3 = arr[0,PART_NAMES["rightShoulder"],1]
    
    dist1 = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    dist2 = math.sqrt((x2-x3)**2 + (y2-y3)**2)
    dist3 = math.sqrt((x1-x3)**2 + (y1-y3)**2)
    '''
    print('X1: '+str(x1))
    print('X2: '+str(x2))
    print('X3: '+str(x3))
    print('Y1: '+str(y1))
    print('Y2: '+str(y2))
    print('Y3: '+str(y3))
    print(dist1)
    print(dist2)
    print(dist3)'''
    cosine=0
    if (2*dist1*dist2)!=0:
        cosine = (dist1**2 + dist2**2 - dist3**2)//(2*dist1*dist2)
    else:
        pass
    return math.acos(cosine)*(180/math.pi) 


def print_armstate(x,arm):
    if(arm==0):
        if(x==0):
            print("left: pause")
        elif(x==1):
            print("left: up")
        elif(x==2):
            print("left: down")
    elif(arm==1):
        if(x==0):
            print("right: pause")
        elif(x==1):
            print("right: up")
        elif(x==2):
            print("right: down")
'''
PART_NAMES = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
]'''

def check_speed():
    if(DATA["avg_left_speed"]>20):
        print("Move your left hand slowly")
    if(DATA["avg_right_speed"]>20):
        print("MOve your right hand slowly")



def main():
    with tf.compat.v1.Session() as sess:
        # prev_keypoint_scores = [0]*5
        # prev_pos_scores = [0]*5
        
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

##        if args.file is not None:
##             cap = cv2.VideoCapture(args.file)
##             
##        else:
        cap = cv2.VideoCapture(args.cam_id)

        #cap = cv2.VideoCapture("C:\\Users\deepg\Desktop\dotslash3-master\posenet-python\side_50.mp4")
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        start = time.time()
        frame_count = 0
        prev_left_angle = 0
        prev_right_angle = 0
        movingarr_size = 20
        movingarr_index = 0
        movingarr_left_angle = np.zeros(movingarr_size)
        movingarr_right_angle = np.zeros(movingarr_size)
        temp1 = 0
        temp2 = 0
        init_param()
        
        while True:
            print("hiin")
            start_iter = time.time()
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)
            #cv2.imshow('test',input_image)


            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=1,
                min_pose_score=0.15)

            keypoint_coords *= output_scale
            #print('key: '+str(keypoint_coords))
            # TODO this isn't particularly fast, use GL for drawing and display someday...
            #print(pose_scores)
            #print(keypoint_scores)
            #print(keypoint_coords)
            #print("left wrist")
            #print(keypoint_coords[0,9,0])
            #print("left angle:")
            
            left_angle = get_angle(keypoint_coords,0)
            right_angle = get_angle(keypoint_coords,1)
            
            DATA["left_angle"] = lowpass(left_angle,movingarr_left_angle,movingarr_index,movingarr_size)
            DATA["right_angle"] = lowpass(right_angle,movingarr_right_angle,movingarr_index,movingarr_size)
            print(DATA["left_angle"])
            #print(right_angle)
            movingarr_index+=1
            update_max()
            update_min()
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)
            cv2.imshow('posenet', overlay_image)
            frame_count += 1
            DATA["left_speed"] = (DATA["left_angle"] - prev_left_angle)
            DATA["right_speed"] = (DATA["right_angle"] - prev_right_angle)
            
            DATA["left_speed"]*=10
            DATA["right_speed"]*=10
            #print(DATA["left_speed"])
            #print(MIN,MAX)
            temp1= temp1 + abs(DATA["left_speed"])
            temp2= temp2 + abs(DATA["right_speed"])
            DATA["avg_left_speed"] = 0.3*temp1/movingarr_index + 0.7*abs(DATA["left_speed"])
            DATA["avg_right_speed"] = 0.3*temp2/movingarr_index + 0.7*abs(DATA["right_speed"])
            prev_left_angle = DATA["left_angle"]
            prev_right_angle = DATA["right_angle"]
            NORM["left_angle"] = (DATA["left_angle"] - MIN["left_angle"])/MAX["left_angle"]
            NORM["right_angle"] = (DATA["right_angle"] - MIN["right_angle"])/MAX["right_angle"]
            #print("avg speed:" + str(DATA["avg_left_speed"]))
            #print(NORM)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            #print("avg left speed: " + str(DATA["avg_left_speed"]))
            check_speed()
            fsm_arm(present_state["left_arm"],0)
            fsm_arm(present_state["right_arm"],1)
            print_armstate(present_state["left_arm"],0)
            print_armstate(present_state["right_arm"],1)
            print_armstate(present_state["right_arm"],1)
            
        print("hi")
        print('Average FPS: ', frame_count / (time.time() - start))
        print(MAX)
        print(MIN)

if __name__ == "__main__":
    main()
