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

 
os.environ['KMP_DUPLICATE_LIB_OK']='True'


                         
def FindFaces(imagesCount):
    
    detectedFacesCount = 0
    detectedFacesLoc = []  
    cnn_face = dlib.cnn_face_detection_model_v1('./detection_model/mmod_human_face_detector.dat')
    for loadedImages in tqdm(range (0, imagesCount)): 
        img = face_recognition.load_image_file('./temp/image{}.png'.format(loadedImages))
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
        try:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
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
                    
            
        for f in detectedFacesLoc:
            if (f[5] > 1):              
                if (f[4] == 1) :
                    cv2.rectangle(img, (f[0],f[1]), (f[2],f[3]), (255,0,0), 2)
                elif (f[4] == 2) :
                   cv2.rectangle(img, (f[0],f[1]), (f[2],f[3]), (0,255,0), 2)                
                else:
                   cv2.rectangle(img, (f[0],f[1]), (f[2],f[3]), (0,0,255), 2)
            else:
                detectedFacesLoc.remove(f)
            
        cv2.imwrite('./detection_output/face_image_{}.jpg'.format(loadedImages),img)
        detectedFacesCount += len(detectedFacesLoc)
    return detectedFacesCount,detectedFacesLoc


# Starting Point of the code




def detect_face_points(image,face_rect):
    #detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("attention_model/face_points_model.dat")
    #face_rect = detector(image, 1)
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

imagesCount=0
images = []
im = cv2.imread('testing_dataset/IMG_8604.png', cv2.IMREAD_COLOR)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
h,w,c = im.shape
if h > 1280 and w > 720:
    im=cv2.resize(im, dsize=(1280,720), interpolation=cv2.INTER_CUBIC)
images.append(im)
cv2.imwrite('./temp/image{}.png'.format(imagesCount),im)
imagesCount+=1
print("\nloaded images successfully")
          


# call "FindFaces()" function and print the number of faces detected using different classifiers
detectedFacesCount, detectedFacesLoc = FindFaces(imagesCount) 
print("\ndetected", detectedFacesCount, "faces")
print(detectedFacesLoc)


face_rect=dlib.rectangles()
for detected_faces in (detectedFacesLoc):
    detected_faces = np.array(detected_faces)[0:-2]

    detected_faces = dlib.rectangle(detected_faces[0], detected_faces[1], 
                                  detected_faces[2], detected_faces[3])
    face_rect.append(detected_faces)

print("-----------------------")
print("Start Face pose model")


total_face_points = detect_face_points(im, face_rect)
print("finished detect_face_points")

total_features = compute_features(total_face_points)
print("finished compute_features")
std = pkl.load(open('./attention_model/std_scaler.pkl', 'rb'))
model = load_model('./attention_model/face_pose_model.h5')
total_poses=[]

labels=[0,1,2,3,4]
uniques, ids = np.unique(labels, return_inverse=True)

for features in total_features:
    features = std.transform(features)
    y_pred = model.predict(features)
    predicted_label=uniques[y_pred.argmax(1)]
    total_poses.append(predicted_label)
    
total_poses = np.array(np.squeeze(total_poses))
print(total_poses)

#5 categories of attention 
attention = [0,0,0,0,0]

'''
im = cv2.imread("testing_dataset/IMG_8604.png")
if h > 1280 and w > 720:
    im=cv2.resize(im, dsize=(1280,720), interpolation=cv2.INTER_CUBIC)
'''
    
#detect faces using FER() , detect emotions 
#then compare detected faces from FER() with detected faces from our classifier
detector = FER()    
students = detector.detect_emotions(im)

for s in students:
    flag = False
    for f, p in zip(detectedFacesLoc,total_poses):
        
        if(f[0]-10 < s["box"][0] < f[0]+10):
            flag = True
            s["pose"] = p
            s["box"] = f
            
    if(not(flag)):
        students.remove(s)
    s["emotions"] = dict(sorted(s["emotions"].items(), key=lambda item: item[1]))
    s["emotions"] = list(s["emotions"].items())
    if(s["emotions"][-1][0] == "happy" and s["pose"] > 1):
        attention[4] +=1  
        s["emotions"] = 1
    elif(s["emotions"][-1][0] == "neutral" and s["pose"] > 1):
        attention[3] +=1  
        s["emotions"] = 0
    elif(s["pose"] > 1):
        attention[2] +=1  
        s["emotions"] = -1
    elif(s["emotions"][-1][0] == "happy" and s["pose"] == 1):
        attention[1] +=1 
        s["emotions"] = 1
    elif(s["emotions"][-1][0] == "neutral" and s["pose"] == 1):
        attention[1] +=1 
        s["emotions"] = 0
    elif(s["emotions"][-1][0] == "neutral" and s["pose"] < 1):
        attention[0] +=1 
        s["emotions"] = 0   
    elif(s["emotions"][-1][0] == "happy" and s["pose"] < 1):
        attention[0] +=1 
        s["emotions"] = 1   
    else:
        attention[0] +=1 
        s["emotions"] = -1   

    
    print(s["emotions"])
    #s["attention"] = []
    #s["attention"].append()

print(attention)
print(students)
plt.imshow(im)
