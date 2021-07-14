import face_recognition
import cv2
import pickle
from imutils import paths
from pathlib import Path
import cvlib as cv
import dlib 

def FindFaces(img):
    
    detectedFacesLoc = [] 
    cnn_face = dlib.cnn_face_detection_model_v1('./detection_model/mmod_human_face_detector.dat')
    
    face_locations =cnn_face(img,0)
        
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
  
                
    if (detectedFacesLoc[0][5] > 1 or len(detectedFacesLoc)== 1): 
        face_crop=img[detectedFacesLoc[0][1]: detectedFacesLoc[0][3],
                          detectedFacesLoc[0][0]:detectedFacesLoc[0][2]]             
    else:
        detectedFacesLoc.remove(detectedFacesLoc[0])
        face_crop=img  
  
    return face_crop, detectedFacesLoc


#get paths of each file in folder named Images
#Images here contains my data(folders of various persons)
imagePaths = list(paths.list_images('./augmented_cropped_images'))
knownEncodings = []
knownNames = []
# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    name = (Path(imagePath).stem).split(sep=" ")[0]
    # to dlib ordering (RGB)
    image = cv2.imread(imagePath)
    im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h,w,c = im_rgb.shape
    if h > 1280 and w > 720:
        im_rgb=cv2.resize(im_rgb, dsize=(1280,720), interpolation=cv2.INTER_CUBIC)
        
    detect_face_img , face_location= FindFaces(im_rgb);
    #Use Face_recognition to locate faces
    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(detect_face_img)
    if (len(encodings)==0 and len(face_location)!=0):
        new_img = im_rgb[face_location[0][1] - 20: face_location[0][3] + 20,
                                  face_location[0][0] - 10:face_location[0][2] + 10]
        encodings = face_recognition.face_encodings(new_img)
        
    if len(encodings)==0:
        encodings = face_recognition.face_encodings(im_rgb)
        
    for encoding in encodings:
        if len(encoding) !=0:
            print("Done image of ", i)
            knownEncodings.append(encoding)
            knownNames.append(name)


#save emcodings along with their names in dictionary data
data ={"encodings":knownEncodings, "names": knownNames}

#use pickle to save data into a file for later use
f = open("face_encoding", "wb")

f.write(pickle.dumps(data))

f.close()
    
# writing to csv file 
'''with open("face_encoding.csv", 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the fields 
    csvwriter.writerow(knownNames) 
        
    # writing the data rows 
    csvwriter.writerow(knownEncodings)'''
    

