Code.
import numpy as np
import cv2
with_mask = np.load('with_mask1.npy')
without_mask = np.load('without_mask1.npy')
with_mask.shape
without_mask.shape
with_mask =with_mask.reshape(200, 50 * 50 * 3)
without_mask =without_mask.reshape(200, 50 * 50 * 3)
with_mask.shape
without_mask.shape
x = np.r_[with_mask , without_mask]
labels = np.zeros(x.shape[0])
labels[200 :] = 1.0
names = {0 : 'MASK', 1 : 'NO MASK'}
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size = 0.25)
x_train.shape
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
x_train = pca.fit_transform(x_train)
x_train[0]
x_train.shape
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size = 0.25)
svm = SVC()
svm.fit(x_train, y_train)
y_pred=svm.predict(x_test)
accuracy_score(y_test, y_pred)
haar_data = cv2.CascadeClassifier('D:/data/harcascade/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
d=[]
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    flag, img = cap.read()
    if flag :   
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face,(50,50))
            face = face.reshape(1,-1)
            pred = svm.predict(face)[0]
            n = names[int(pred)]
            cv2.putText(img,n,(x,y),font,1,(244,250,255),2)
            print(n)
        cv2.imshow('img',img)
        if cv2.waitKey(2) == 27:
            break
            
cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
img1=cv2.imread('C:/Users/DELL/Pictures/Camera Roll/pic1.jpg')
img2=cv2.imread('C:/Users/DELL/Pictures/Camera Roll/pic2.jpg')
diff=cv2.subtract(img1,img2)
result = not np.any(diff)
if result is True:
    print ("Good to Go")
else:
    cv2.imwrite("result.jpg",diff)
    print ("Challan has been generated")
    s="fine is generated"
    with open('E:\smart\prash.csv','r+') as f:
        f.writelines({s})
