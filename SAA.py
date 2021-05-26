import cv2 
from tqdm import tqdm
import cvlib as cv
import face_recognition
import dlib 
import os
import numpy as np
import _pickle as pkl
from keras.models import load_model
from fer import FER
import matplotlib.pyplot as plt
import time
from enum import Enum
import copy
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Pose_name(Enum):
    right = 0
    slightly_right = 1
    center = 2
    slightly_left = 3
    left = 4

class Student:

    
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.name = ''
        self.poses = []
        self.emotions = []
    
    def set_coordinates(self, coordinates):
        self.coordinates = coordinates
    
    def set_name(self,name):
        self.name = name
        
    def add_pose(self, pose):
        self.poses.append(pose)
        
    def add_emotion(self, emotion):
        self.emotions.append(emotion)
    
    def get_coordinates(self):
        return self.coordinates
    
    def get_name(self):
        return self.name
    
    def get_poses(self):
        return self.poses
    
    def get_emotions(self):
        return self.emotions
   
    def print_student(self):
        print("Coordinates: {}".format(self.coordinates))
        print("Name: {}".format(self.name))
        print("Poses: {}".format(self.poses))
        print("Emotions: {}".format(self.emotions))
        
####################### System Functions #####################################       
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h,w,c = img.shape
    if h > 1280 and w > 720:
        img=cv2.resize(img, dsize=(1280,720), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('./temp/image{}.png'.format(number_of_images),img)
    return img

def FindFaces(img):
    detectedFacesLoc = []  
    
    ###################### CNN Classifier ######################
    cnn_face = dlib.cnn_face_detection_model_v1('./detection_model/mmod_human_face_detector.dat')
    face_locations =cnn_face(img,0)
    if (len(face_locations) == 0):
        face_locations =cnn_face(img,1)
    if (len(face_locations) > 0):  
        for face in face_locations:
            x1 = face.rect.left()
            y1 = face.rect.top()
            x2 = face.rect.right()
            y2 = face.rect.bottom()
            c = 1
            count = 1
            detectedFacesLoc.append([x1,y1,x2,y2,c,count])
    
    ###################### Face Recoginition Classifier ######################
    try:
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        print("Something went wrong")
    finally:
        face_locations = face_recognition.face_locations(grayscale_image)
        
    if (len(face_locations) > 0):  
        for face in face_locations:
            y1,x2,y2,x1 = face
            found = 0
            for dface in detectedFacesLoc:
                if (x1 < dface[0] + 30 and x1 > dface[0] - 30):
                    found = 1
                    dface[5] += 1
                    break
            if (found == 0):
                c = 2
                detectedFacesLoc.append([x1,y1,x2,y2,c,1])
         
    ###################### CV Lib Classifier ######################                 
    face_locations, confidences = cv.detect_face(img)
    if (len(face_locations) > 0):  
        for face in face_locations:
            x1,y1,x2,y2 = face
            found = 0
            for dface in detectedFacesLoc:
                if (x1 < dface[0] + 30 and x1 > dface[0] - 30):
                    found = 1
                    dface[5] += 1
                    break
            if (found == 0):
                c = 3
                detectedFacesLoc.append([x1,y1,x2,y2,c,1])
                    
    ################## Draw Rectangles ############################        
    for f in detectedFacesLoc:
        if (f[5] > 1):              
            cv2.rectangle(img, (f[0],f[1]), (f[2],f[3]), (0,0,255), 2)
        else:
            detectedFacesLoc.remove(f)
            
    cv2.imwrite('./detection_output/face_image_{}.jpg'.format(number_of_images),img)
    return detectedFacesLoc

def write_output(img, student_dict, mode):
    temp_img = copy.deepcopy(img)
    for student in student_dict:
        if mode =='pose':
            text_to_display = Pose_name(student[mode]).name
        else:
            text_to_display = student[mode]
        location = student['box']
        cv2.rectangle(temp_img, (location[0],location[1]), (location[2],location[3]), (255,0,0), 2)
        cv2.putText(temp_img, text_to_display, (location[0] + 10, location[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
            
    cv2.imwrite('./{}_output/face_image_{}.jpg'.format(mode, number_of_images),temp_img)
    return img

def to_rectangles(detectedFacesLoc):
    face_rect=dlib.rectangles()
    for detected_faces in (detectedFacesLoc):
        detected_faces = np.array(detected_faces)[0:-2]
        detected_faces = dlib.rectangle(detected_faces[0], detected_faces[1], detected_faces[2], detected_faces[3])
        face_rect.append(detected_faces)
    return face_rect

'''
The get_pose() function uses the two functions {detect_face_points, compute_feature} 
to get the poses of each face
'''

def get_pose(img, face_rect):
    total_poses=[]
    total_face_points = detect_face_points(img, face_rect)
    total_features = compute_features(total_face_points)
    labels = [0,1,2,3,4]
    uniques, ids = np.unique(labels, return_inverse=True)
    std = pkl.load(open('./attention_model/std_scaler.pkl', 'rb'))
    model = load_model('./attention_model/face_pose_model.h5')
    for features in total_features:
        features = std.transform(features)
        y_pred = model.predict(features)
        predicted_label=uniques[y_pred.argmax(1)]
        total_poses.append(predicted_label)
    total_poses = np.array(np.squeeze(total_poses))
    return total_poses


def detect_face_points(image,face_rect):
    predictor = dlib.shape_predictor("attention_model/face_points_model.dat")
    total_face_points=[]
    for rect in face_rect:
        face_points=[]
        dlib_points = predictor(image, rect)
        for i in range(68):
            x, y = dlib_points.part(i).x, dlib_points.part(i).y
            face_points.append(np.array([x, y]))
        total_face_points.append(face_points)
    return total_face_points


def compute_features(total_face_points):
    total_features=[]
    for face_points in total_face_points:
        face_points = np.array(face_points)
        features = []
        for i in range(68):
            for j in range(i+1, 68):
                features.append(np.linalg.norm(face_points[i]-face_points[j]))
        features=np.array(features).reshape(1, -1)
        total_features.append(features)    
    return total_features


def compare_and_set_pose(detectedFacesLoc, students_dict, total_poses):
    temp_dict=[]
    
    for student in students_dict:
        for face, pose in zip(detectedFacesLoc,total_poses):
            
            if(face[0]-20 <= student["box"][0] <= face[0]+20):
                student['box'] = face[:-2]
                student["pose"] = pose
                temp_dict.append(student)

    return temp_dict

def set_emotion(students_dict):
    #attention = [0,0,0,0,0]

    for student in students_dict:
        student["emotions"] = dict(sorted(student["emotions"].items(), key=lambda item: item[1]))
    
        student["emotions"] = list(student["emotions"].items())
        
        student["emotions"] = student["emotions"][-1][0]
        '''
        if(dominant_emotion == "happy" and student_pose > 1):
            attention[4] +=1  
            student["emotions"] = 1
        elif(dominant_emotion == "neutral" and student_pose > 1):
            attention[3] +=1  
            student["emotions"] = 0
        elif(student_pose):
            attention[2] +=1  
            student["emotions"] = -1
        elif(dominant_emotion == "happy" and student_pose == 1):
            attention[1] +=1 
            student["emotions"] = 1
        elif(dominant_emotion == "neutral" and student_pose == 1):
            attention[1] +=1 
            student["emotions"] = 0
        elif(dominant_emotion == "neutral" and student_pose < 1):
            attention[0] +=1 
            student["emotions"] = 0   
        elif(dominant_emotion == "happy" and student_pose < 1):
            attention[0] +=1 
            student["emotions"] = 1   
        else:
            attention[0] +=1 
            student["emotions"] = -1
        '''
    return students_dict

def fill_students(students_dict, students_list):
    if len(students_list)== 0:
        for s in students_dict:
            students_list.append(Student(s['box']))
            
    for student_object in  students_list:
        student_top_left = student_object.get_coordinates()[0]
        for student in students_dict:
            if ( student['box'][0] - 20 < student_top_left < student['box'][0] + 20):    
                student_object.add_pose(student['pose'])
                student_object.add_emotion(student['emotions'])
    


def main():
    global number_of_images 
    number_of_images = 0
    start_time = time.time()
    path_list = ['testing_dataset/IMG_8607.png','testing_dataset/IMG_8608.png']
    print("Starting the system.")
    for path in path_list:
    
        print('-' * 40)
        img = load_image(path)
        print("Loaded Image Successfully.")
        
        print('-' * 40)
        detectedFacesLoc = FindFaces(img) 
        print("Face Locations: ", detectedFacesLoc)
        
        print('-' * 40)
        face_rect = to_rectangles(detectedFacesLoc)
        print("Face Rectangles: ", face_rect)
        
        print('-' * 40)
        total_poses = get_pose(img, face_rect)
        print("Students' poses: ", total_poses)
        
        print('-' * 40)
        detector = FER(mtcnn=True)    
        students_dict = detector.detect_emotions(img)
        print("Students' Dictionary: \n", students_dict)
        
        print('-' * 40)
        students_dict = compare_and_set_pose(detectedFacesLoc, students_dict, total_poses)
        print("Students Dictionary after removing the incorrect faces (if exists) and setting the pose of each student: \n", students_dict)
        img = write_output(img, students_dict, 'pose')

        print('-' * 40)
        students_dict = set_emotion(students_dict)
        print("Students dictionary after adding the dominant emotion for each student: \n",students_dict)
        
        img = write_output(img, students_dict, 'emotions')

        print('-' * 40)
        fill_students(students_dict, students_list)
        
        number_of_images +=1
        
    execution_time = time.time() - start_time
    for student_object in  students_list:    
        student_object.print_student()
        print('#'*40)
    print('-' * 40)
    print("System Terminated Successfully in {} sec ".format(execution_time))
    
############################# Start of code ###################################
students_list =[]
main()

