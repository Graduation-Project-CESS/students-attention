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
import pickle 

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Pose_name(Enum):
    right = 0
    slightly_right = 1
    center = 2
    slightly_left = 3
    left = 4

class Student:

    
    def __init__(self, coordinates, name):
        self.coordinates = coordinates
        self.name = name
        self.poses = []
        self.emotions = []
    
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
    temp_img = copy.deepcopy(img)
     
    for f in detectedFacesLoc:
        if (f[5] > 1):              
            cv2.rectangle(temp_img, (f[0],f[1]), (f[2],f[3]), (0,0,255), 2)
        else:
            detectedFacesLoc.remove(f)
            
    cv2.imwrite('./detection_output/face_image_{}.jpg'.format(number_of_images),temp_img)
    return detectedFacesLoc

'''
def face_recog(detectedFacesLoc, image):
    
    data=pickle.loads(open("face_encoding", 'rb').read())
    locations=[]
    names = []
    
    #for encoding in encodings:
    for Loc in detectedFacesLoc:
        img_crop=image[Loc[1]: Loc[3],Loc[0]:Loc[2]]
        encodings=face_recognition.face_encodings(img_crop)
        if len(encodings)==0:
            img_crop=cv2.resize(img_crop, dsize=(180,180), interpolation=cv2.INTER_CUBIC)
            encodings=face_recognition.face_encodings(img_crop)
        for encoding in encodings:
            print(encodings[0])
            print('---------')
            print(encoding)
            print('11111111111111111111111111111111111')
            return
            break
            print('-'*50)
            match_count=0
            matches = face_recognition.compare_faces(data['encodings'],encoding,tolerance=0.6)      
            ##print(matches)
            #set name =unknown if no encoding matches
            name = "Unknown" 
            #matches=np.array(matches)
            for m in matches:
                if m :
                    match_count+=1
            # check to see if we have found a match
            if match_count > 5 :
                #Find positions at which we get True and store them
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    #Check the names at respective indexes we stored in matchedIdxs
                    name=data['names'][i]
                    #name.append(knownnames[i])
                    #increase count for the name we got
                    counts[name] = counts.get(name, 0) + 1
                    #set name which has highest count
                    name = max(counts, key=counts.get)
                ##print(name)
        # update the list of names
            names.append(name)
            locations.append(Loc)
    
    print(names)
    # loop over the recognized faces
    for (loc, name) in zip(locations , names):
        # draw the predicted face name on the image
        cv2.putText(image, name, (loc[0],loc[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.imwrite("./recognition_output/rec_img.png",image)
'''
def face_recog(detectedFacesLoc, img):
    
    data=pickle.loads(open("face_encoding", 'rb').read())
    names = []

    #for encoding in encodings:
    for i, Loc in enumerate(detectedFacesLoc):
        img_crop=img[Loc[1]: Loc[3],Loc[0]:Loc[2]]
        encodings=face_recognition.face_encodings(img_crop)
        if len(encodings)==0:
            img_crop=img[Loc[1] - 20: Loc[3] + 20,Loc[0] - 10:Loc[2] + 10]
            encodings=face_recognition.face_encodings(img_crop)
        cv2.imwrite("./cropped_faces/crop_img_{}_{}.jpg".format(number_of_images, i),img_crop) 
        match_count=0
        name = "Unknown" 
        try:
            matches = face_recognition.compare_faces(data['encodings'],encodings[0],tolerance=0.6)      
    
            #set name =unknown if no encoding matches
            #matches=np.array(matches)
            for m in matches:
                if m :
                    match_count+=1
            # check to see if we have found a match
            if match_count > 5 :
                #Find positions at which we get True and store them
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    #Check the names at respective indexes we stored in matchedIdxs
                    name=data['names'][i]
                    #name.append(knownnames[i])
                    #increase count for the name we got
                    counts[name] = counts.get(name, 0) + 1
                    
                    #set name which has highest count
                name = max(counts, key=counts.get)
                ##print(name)
        except:
            print('?')                            
        # update the list of names
        names.append(name)
    
    temp_img = copy.deepcopy(img)
    # loop over the recognized faces
    for (loc, name) in zip(detectedFacesLoc , names):
        # draw the predicted face name on the image
        cv2.putText(temp_img, name, (loc[0],loc[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.imwrite("./recognition_output/rec_img_{}.jpg".format(number_of_images),temp_img)
    
    return names

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


def compare_and_set_pose(detectedFacesLoc, students_dict, total_poses, names):
    temp_dict=[]
    
    for s in students_dict:
        for face, pose, name in zip(detectedFacesLoc,total_poses, names):
            
            if ((face[0]-50 <= s["box"][0] <= face[0]+50) 
            and (face[1]-50 <= s["box"][1] <= face[1]+50)
            and (face[2]-50 <= s["box"][0] + s["box"][2] <= face[2]+50)
            and (face[3]-50 <= s["box"][1] + s["box"][3] <= face[3]+50)):
                s['box'] = face[:-2]
                s["pose"] = pose
                s["name"] = name
                temp_dict.append(s)

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
    #Assume all students will appear in the first image.
    if len(students_list)== 0:
        for s in students_dict:
            students_list.append(Student(s['box'], s['name']))
            
            
    for s in students_dict:  
        flag = False
        for student_object in  students_list:
            student_name = student_object.get_name()
            student_top_left_x = student_object.get_coordinates()[0]
            student_top_left_y = student_object.get_coordinates()[1]
            student_bottom_right_x = student_top_left_x + student_object.get_coordinates()[2]
            student_bottom_right_y = student_top_left_y + student_object.get_coordinates()[3]
            if (student_name == s['name'] or
                ((s["box"][0]-50 <= student_top_left_x <= s["box"][0]+50) 
            and (s["box"][1]-50  <= student_top_left_y <= s["box"][1]+50)
            and (s["box"][2]-50  <= student_bottom_right_x <= s["box"][2]+50)
            and (s["box"][3]-50  <= student_bottom_right_y <= s["box"][3]+50))):    
                student_object.add_pose(s['pose'])
                student_object.add_emotion(s['emotions'])
                flag = True
        if ((not flag) and s['name']!= "Unknown"):
            new_student = Student(s['box'], s['name'])
            new_student.add_pose(s['pose'])
            new_student.add_emotion(s['emotions'])
            students_list.append(new_student)

def get_time(seconds):
    seconds_in_hour = 60 * 60
    seconds_in_minute = 60

    hours = seconds // seconds_in_hour
    minutes = (seconds  - (hours * seconds_in_hour)) // seconds_in_minute
    seconds = seconds % minutes    
    print("System Terminated Successfully :{} hours, {} minutes, {} seconds".format(hours, minutes, seconds))

def main():
    global number_of_images 
    number_of_images = 0
    start_time = time.time()
    path_list = ['testing_dataset/IMG_8607.png',
                 'testing_dataset/IMG_8608.png',
                 'testing_dataset/IMG_8609.png',
                 'testing_dataset/IMG_8610.png',
                 'testing_dataset/IMG_8611.png',
                 'testing_dataset/IMG_8612.png',
                 'testing_dataset/IMG_8613.png',
                 'testing_dataset/IMG_8614.png',]
    print("Starting the system.")
    for path in path_list:
    
        print('-' * 40)
        img = load_image(path)
        print("Loaded Image Successfully.")
        
        print('-' * 40)
        detectedFacesLoc = FindFaces(img) 
        print("Face Locations: ", detectedFacesLoc)
        
        print('-' * 40)
        names = face_recog(detectedFacesLoc, img)
        print("Face Names: ", names)
        
        print('-' * 40)
        face_rect = to_rectangles(detectedFacesLoc)
        print("Face Rectangles: ", face_rect)
        
        print('-' * 40)
        total_poses = get_pose(img, face_rect)
        print("Students' poses: ", total_poses)
        
        print('-' * 40)
        detector = FER(mtcnn=True)   #Using MT CNN for face detection to improve accuracy 
        students_dict = detector.detect_emotions(img)
        print("Students' Dictionary: \n", students_dict)
        
        print('-' * 40)
        students_dict = compare_and_set_pose(detectedFacesLoc, students_dict, total_poses, names)
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
    get_time(execution_time)
    
############################# Start of code ###################################
students_list =[]
main()

