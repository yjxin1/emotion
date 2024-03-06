from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import time
from matplotlib import pyplot as plt
from lenet5 import LeNet
import cv2
import numpy as np

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('emotion/models/model-mine.pth')  # 加载模型
    model = model.to(device)
    model.eval()  # 把模型转为test模式
    trans = transforms.Compose(
        [
            # 将图片尺寸resize到32x32
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # 0.1307, 0.3081是统计出来的
        ])
    
   # 实例化级联分类器,加载分类器
    face_path = 'emotion/face/haarcascade_frontalface_default.xml'
    face_cas = cv2.CascadeClassifier(face_path)
    face_cas.load(face_path)
   
    cap = cv2.VideoCapture("emotion/face/a.mp4")   # 从视频文件捕获视频
    while True:
    # 读取视频帧
        ret, frame = cap.read()

        if ret:
            img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faceRects = face_cas.detectMultiScale(
            img, scaleFactor=1.2, minNeighbors=4, minSize=(10, 10))
            img_addROI = frame
            for faceRect in faceRects:
                x, y, w, h = faceRect
                # 框出人脸
                img_addROI = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                faceRect=frame[y:y+h,x:x+w]
                faceRect=TF.to_pil_image(faceRect)
                faceRect=faceRect.convert('L')
                face=trans(faceRect)
                face = face.to(device)
                face = face.unsqueeze(0)
                output = model(face)
                prob = F.softmax(output, dim=1)
                value, predicted = torch.max(output.data, 1)
                predict = output.argmax(dim=1)
                a=predict.item()
                value= round(torch.max(prob).item(),3)
                #print(a)
                if a==0:
                    emo=cv2.imread('emotion/qqemo/Angry.png')
                    st='angry '
                elif a==1:
                    emo=cv2.imread('emotion/qqemo/Disgust.png')
                    st='disgust '
                elif a==2:
                    emo=cv2.imread('emotion/qqemo/Fear.png')
                    st='fear '
                elif a==3:
                    emo=cv2.imread('emotion/qqemo/Happy.png')
                    st='happy '
                elif a==4:
                    emo=cv2.imread('emotion/qqemo/Sad.png')
                    st='sad '
                elif a==5:
                    emo=cv2.imread('emotion/qqemo/Surprise.png')
                    st='surprise '
                elif a==6:
                    emo=cv2.imread('emotion/qqemo/Neutral.png')
                    st='neutral '
                emo=cv2.resize(emo,(100,100))
                img_addROI[0:100,0:100,:]=emo
                cv2.putText(img_addROI, st + str(value), (int((x+w)/2), int((y+h)/2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)   
        cv2.imshow('face', img_addROI)
        if cv2.waitKey(60) & 0xFF == 27:
            break  # 结束当前循环
    cv2.destroyAllWindows()
