import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pkl
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import  backend as K


x, y = pkl.load(open('training_attention/samples.pkl', 'rb'))
y_new = []
for i in range(len(y)):
    if (y[i][2] < -50):
        y[i] = -2
    elif (y[i][2] > -50 and y[i][2] < -10):
        y[i] = -1    
    elif (y[i][2] > -10 and y[i][2] < 10):
        y[i] = 0     
    elif (y[i][2] > 10 and y[i][2] < 50):
        y[i] = 1     
    else:
        y[i] = 2 
    y_new.append(y[i][0])


x_train, x_test, y_train, y_test = train_test_split(x, y_new, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)


std = StandardScaler()
std.fit(x_train)
pkl.dump(std, open('attention_model/std_scaler.pkl','wb'))
x_train = std.transform(x_train)
x_val = std.transform(x_val)
x_test = std.transform(x_test)

BATCH_SIZE = 64
EPOCHS = 100
Learning_rate=0.001


model = Sequential()
model.add(Dense(units=20, activation='relu', kernel_regularizer='l2', input_dim=x.shape[1]))
model.add(Dense(units=10, activation='relu', kernel_regularizer='l2'))
model.add(Dense(units=30, activation='relu', kernel_regularizer='l2'))
model.add(Dense(units=1, activation='linear'))

print(model.summary())

es = EarlyStopping(monitor='accuracy', mode='max', verbose=1,patience=20)
#set model checkpoint to save the model whenever getting accuracy higher than current max one
mc = ModelCheckpoint('attention_model/face_pose_model.h5', monitor='accuracy', mode='max',verbose=1, save_best_only=True)

model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
K.set_value(model.optimizer.learning_rate, Learning_rate)
hist = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[es,mc])

print('Train loss:', model.evaluate(x_train, y_train, verbose=0))
print('  Val loss:', model.evaluate(x_val, y_val, verbose=0))
print(' Test loss:', model.evaluate(x_test, y_test, verbose=0))

history = hist.history
loss_train = history['loss']
loss_val = history['val_loss']


plt.figure()
plt.plot(loss_train, label='train')
plt.plot(loss_val, label='val_loss', color='red')
plt.legend()




def detect_face_points(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("attention_model/face_points_model.dat")
    face_rect = detector(image, 1)
    print(type(face_rect))
    print(face_rect)
    if len(face_rect) != 1: return []
    print(face_rect[0])
    dlib_points = predictor(image, face_rect[0])
    face_points = []
    for i in range(68):
        x, y = dlib_points.part(i).x, dlib_points.part(i).y
        face_points.append(np.array([x, y]))
    cv2.imwrite('./face_detection_images/face_image_1.png',image)
    return face_points
        
def compute_features(face_points):
    assert (len(face_points) == 68), "len(face_points) must be 68"
    
    face_points = np.array(face_points)
    features = []
    for i in range(68):
        for j in range(i+1, 68):
            features.append(np.linalg.norm(face_points[i]-face_points[j]))
            
    return np.array(features).reshape(1, -1)

im = cv2.imread('testing_dataset/IMG_8589_1.png', cv2.IMREAD_COLOR)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

face_points = detect_face_points(im)

for x, y in face_points:
    cv2.circle(im, (x, y), 1, (0, 255, 0), -1)
    
features = compute_features(face_points)
std = pkl.load(open('attention_model/std_scaler.pkl', 'rb'))
features = std.transform(features)

model = load_model('attention_model/face_pose_model.h5')
y_pred = model.predict(features)

print('  Y: ',y_pred)


