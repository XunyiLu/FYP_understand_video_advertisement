import cv2
import os

if __name__ == '__main__':

    basePath = "/Users/lxy/Desktop/FYP/videos/frameExtraction/"
    directory = os.fsencode(basePath)

    for file in os.listdir(directory):
        fileName = os.fsdecode(file)
        fileName = fileName.replace("\n", "")
        # print(basePath + fileName)
        # print(pathway)
        cap = cv2.VideoCapture(basePath + fileName)
        c = 1
        frameRate = 100  # Interval between frames
        
        while (True):
           ret, frame = cap.read()
           if ret:
               if c % frameRate == 0:
                  print("Start video capturing " + str(c) + " frame")
                  cv2.imwrite("/Users/lxy/Desktop/FYP/videos/frameExtraction/" + fileName + str(c) + ".jpg", frame)
               c += 1

           else:
                  print("All frames saved")
                  break
        cap.release()
