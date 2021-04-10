import cv2 
# import json
# import codecs
# import requests
# import shutil
# import glob
# from PIL import Image
from tqdm import tqdm
import cvlib as cv
import face_recognition
import dlib 
import os
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pkl
from keras.models import load_model
# from keras.layers import Dense
# from keras.callbacks import EarlyStopping
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

os.environ['KMP_DUPLICATE_LIB_OK']='True'

'''
#Load images from URL links in the JSON file to the "Images" list & store them locally to be used by the first classifier.
def LoadImages():
    imagesCount=0
    facesCount=0
    for data in tqdm(jsonData[20:30]):

        facesCount += len(data["annotation"])
        response = requests.get(data['content'], stream=True)
        with open('./temp/my_image.jpg', 'wb') as file:
            shutil.copyfileobj(response.raw, file)
        del response
        img = Image.open('./temp/my_image.jpg')   
        img = np.asarray(img)
        cv2.imwrite('./temp/loaded_images/image{}.png'.format(imagesCount),img)
        images.append(img)
        imagesCount +=1
    return imagesCount, facesCount
'''
'''
Detect Faces in the images and save their coordinates in detectedFacesLoc
pass the image to the 3 classifiers sequentially CNN then Face-recognition then CVLib
It's based on 3 different classifiers.
Firstly, use the CNN library, by using APIs provided by it to detect the faces.
then use the second classifier "face_recognition classifier"
Lastly use the last classifier which is implemented from "cvlib" library.

after running first classifier and savig coordinates of all faces detected
we run second classifier and compare all detected faces by second classifier to
the ones by first classifier and if any 2 faces have same coordinates range 
we save one of them but we increase its count(no. of times this face is found)

if there is a face found only 1 time it is not saved
we save only faces that are found more than one time
'''
                         
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
'''
address = './datasets/1.json'

jsonData = []
images = []

#Load URL links to the list jsonData & delete the url link of index (272) due to technical issues
with codecs.open(address, 'rU', 'utf-8') as js:
    for line in js:
        jsonData.append(json.loads(line))

del jsonData[272]


# call "LoadImages()" function & print the total number of images and faces.
imagesCount, facesCount= LoadImages()
print("\n{} images were loaded successfully, Which contains {} faces".format(imagesCount,facesCount))


# delete any content in face-detection images and loaded-images
folders = ['./Face_detection_images/','./temp/loaded_images/']
for folder in folders: 
    if not os.path.exists(folder):
        os.makedirs(folder)
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
'''
            
#path = glob.glob("./images/*.jpg")


def detect_face_points(image,face_rect):
    #detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("attention_model/face_points_model.dat")
    #face_rect = detector(image, 1)
    print(face_rect)
    print(face_rect.tl_corner())
    print(type(face_rect.br_corner()))
    print(face_rect.top())
    top_left = (face_rect.left() , face_rect.top() )
    bottom_right = (face_rect.right(), face_rect.bottom())
    dlib_points = predictor(image, face_rect)
    face_points = []
    for i in range(68):
        x, y = dlib_points.part(i).x, dlib_points.part(i).y
        face_points.append(np.array([x, y]))
    cv2.rectangle(image, top_left, bottom_right, (0,255,0), 2)
    cv2.imwrite('./attention _images/face_image_1.jpg',image)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    return face_points
        
def compute_features(face_points):
    assert (len(face_points) == 68), "len(face_points) must be 68"
    
    face_points = np.array(face_points)
    features = []
    for i in range(68):
        for j in range(i+1, 68):
            features.append(np.linalg.norm(face_points[i]-face_points[j]))
            
    return np.array(features).reshape(1, -1)

imagesCount=0
images = []
path = "testing_dataset/IMG_8589_1.png"
#for img in tqdm(path):  
im = cv2.imread('testing_dataset/IMG_8608.png', cv2.IMREAD_COLOR)
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

# Program terminated successfully without any errors
#accuracy= (detectedFacesCount / facesCount) * 100
#print("Program terminxated successfully with accuracy: {} %".format(accuracy))
print(detectedFacesLoc)
detectedFacesLoc = np.squeeze(np.array(detectedFacesLoc)[: , 0:-2])
detectedFacesLoc = dlib.rectangle(detectedFacesLoc[0], detectedFacesLoc[1], 
                                  detectedFacesLoc[2], detectedFacesLoc[3])

print(type(detectedFacesLoc))
rectangles_detected=dlib.rectangles()
print(type(rectangles_detected))
print(rectangles_detected)
rectangles_detected = rectangles_detected.append(x=detectedFacesLoc)
print(rectangles_detected)
face_points = detect_face_points(im, detectedFacesLoc)

for x, y in face_points:
    cv2.circle(im, (x, y), 1, (0, 255, 0), -1)
    
features = compute_features(face_points)
std = pkl.load(open('attention_model/std_scaler.pkl', 'rb'))
features = std.transform(features)

model = load_model('attention_model/face_pose_model.h5')
y_pred = model.predict(features)

roll_pred, pitch_pred, yaw_pred = y_pred[0]
print(' Roll: {:.2f}°'.format(roll_pred))
print('Pitch: {:.2f}°'.format(pitch_pred))
print('  Yaw: {:.2f}°'.format(yaw_pred))