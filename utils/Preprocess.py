import numpy as np
import cv2

class Preprocess():
    def __init__(self, length=4, size=32):
        self.hist=np.zeros((length,size,size,1))
        self.size=size
        self.length=length

    def reset(self):
        self.hist=np.zeros((self.length,self.size,self.size,1))

    def __call__(self, img):
        resized = cv2.resize(img, (self.size,self.size), interpolation = cv2.INTER_AREA)
        resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        resized = np.divide(resized, 255)
        resized = np.reshape(resized, resized.shape + (1,))
        self.hist = np.delete(self.hist, 0, 0)
        self.hist = np.append(self.hist, [resized], axis=0)
        final_img = np.concatenate(self.hist, axis=0)
        
        return final_img