import cv2
import numpy as np
import copy
import random
import time
from .datasetInfo import *


def getRawAnnosList(annoFilePath, tell=True):
    annos = []
    with open(annoFilePath) as f:
        for line in f:
            annos.append(line.strip().split())
    if tell:
        print("The annotation file's length is: " + str(len(annos)))
    
    return annos


def saveImgWithLandmarksInGetItem(img, pts, name='img'):
    imgRaw = copy.deepcopy(img)
    imgRaw = np.clip(cv2.cvtColor(imgRaw, cv2.COLOR_RGB2BGR) / 255.0, 0., 1.)
    imgRaw = np.clip(np.array(imgRaw * 255., dtype=np.uint8), 0, 255)
    for i in range(kptNum):
        cv2.circle(
            imgRaw,
            (int(pts[i][0]), int(pts[i][1])),
            1,
            (0, 0, 255),
            -1
        )
        # cv2.putText(imgRaw, str(i), (int(pts[i][0]), int(pts[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
    cv2.imwrite('./'+name+'.png', imgRaw)


def randomParam(augment, width, height):
    alpha_, shiftX, shiftY, flip, sigma_, tau_, eta_, gamma_, theta_, angle, beta_, cX, cY = \
        0, 0., 0., 0, 0, 0, 0, 0, 0, 0., 0, 0., 0.
    if augment:
        random.seed(time.time())
        alpha_ = random.randint(0, 1)
        flip = random.randint(0, 1)
        sigma_ = random.randint(0, 1)
        tau_ = random.randint(0, 1)
        eta_ = random.randint(0, 1)
        gamma_ = random.randint(0, 1)
        theta_ = random.randint(0, 1)
        beta_ = random.randint(0, 1)

        shiftX = random.uniform(-width*alpha, width*alpha) if alpha_ else 0.
        shiftY = random.uniform(-height*alpha, height*alpha) if alpha_ else 0.
        angle = random.uniform(-theta, theta) if theta_ else 0.
        cX = random.uniform(-beta, beta) if beta_ else 0.
        cY = random.uniform(-beta, beta) if beta_ else 0.

    shearMat = np.array([[1., cX, 0.], [cY, 1., 0.]], dtype=np.float32)

    return flip, angle, shearMat, shiftX, shiftY, sigma_, tau_, eta_, gamma_


def colorJetting(img, eta_in):
    eta_min = 255 * np.array([random.uniform(0, eta_in), random.uniform(0, eta_in), random.uniform(0, eta_in)])
    eta_max = 255 * np.array([random.uniform(1-eta_in, 1), random.uniform(1-eta_in, 1), random.uniform(1-eta_in, 1)])
    for ch in range(3):
        img[:, :, ch][np.where(img[:, :, ch] < eta_min[ch])] = 0.
        img[:, :, ch][np.where(img[:, :, ch] > eta_max[ch])] = 255.
        img[:, :, ch][np.where((img[:, :, ch] >= eta_min[ch]) & (img[:, :, ch] <= eta_max[ch]))] = \
            255.*(img[:, :, ch][np.where((img[:, :, ch] >= eta_min[ch]) & (img[:, :, ch] <= eta_max[ch]))] -
                  eta_min[ch])/(eta_max[ch]-eta_min[ch])
    return img


def imgOcclusion(img, gamma_in):
    x_loc = random.randint(0, cropSize - 1)
    y_loc = random.randint(0, cropSize - 1)
    w = random.randint(0, int(cropSize * gamma_in))
    h = random.randint(0, int(cropSize * gamma_in))
    up = int(y_loc-h/2) if int(y_loc-h/2) >= 0 else 0
    down = int(y_loc+h/2) if int(y_loc+h/2) <= int(cropSize-1) else int(cropSize-1)
    left = int(x_loc-w/2) if int(x_loc-w/2) >= 0 else 0
    right = int(x_loc+w/2) if int(x_loc+w/2) <= int(cropSize-1) else int(cropSize-1)
    img[up:down, left:right, :] = np.random.randint(0, int(cropSize-1), size=(down-up, right-left, 3))
    img = np.array(img, dtype=np.float32)
    return img


def getItem(isAugment, annoLine, useFlip=True):
    img = cv2.imread(annoLine[-1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)
    pts = np.array(list(map(float, annoLine[:2*kptNum])), dtype=np.float32)
    pts = (np.vstack((pts[:kptNum], pts[kptNum:]))).transpose().reshape(-1, 2)
    ptsRaw = copy.deepcopy(pts)
    imgWidth, imgHeight = img.shape[1], img.shape[0]

    flip, angle, shearMat, shiftX, shiftY, sigma_, tau_, eta_, gamma_ = randomParam(isAugment, imgWidth//2, imgHeight//2)
    
    flip = flip if useFlip else 0
    
    # rotation
    rotMat = cv2.getRotationMatrix2D((imgWidth / 2.0 - 0.5, imgHeight / 2.0 - 0.5), angle, 1)
    img = cv2.warpAffine(img, rotMat, (imgWidth, imgHeight))
    center = np.array([imgWidth / 2.0, imgHeight / 2.0], dtype=np.float32)
    rad = angle*np.pi/180
    rotMat = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]], dtype=np.float32)
    pts = np.matmul(pts-center, rotMat) + center
    
    # shear transform
    img = cv2.warpAffine(img, shearMat, (imgWidth, imgHeight))
    pts = np.matmul(pts, (shearMat[:, :2]).transpose())
    
    # bounding box perturbation and crop
    ptsBefore = np.float32([
        [imgWidth//4 - shiftX, imgHeight//4 - shiftY],
        [imgWidth//4 - shiftX, imgHeight - 1 - imgHeight//4 + shiftY],
        [imgWidth - 1 - imgWidth//4 + shiftX, imgHeight - 1 - imgHeight//4 + shiftY]
    ])
    ptsAfter = np.float32([  # after, in cropped image space
        [0, 0],
        [0, cropSize - 1],
        [cropSize - 1, cropSize - 1]
    ])
    cropMat = cv2.getAffineTransform(ptsBefore, ptsAfter)
    img = cv2.warpAffine(img, cropMat, (cropSize, cropSize))
    pts = np.matmul(pts, (cropMat[:, :2]).transpose()) + cropMat[:, 2:].reshape(-1, 2)
    
    # flip
    img = np.array(np.fliplr(img), dtype=np.float32) if flip else img
    pts[:, 0] = cropSize - pts[:, 0] if flip else pts[:, 0]
    pts = pts[np.array(flipRelation)[:, 1], :] if flip else pts
    pts = np.array(pts, dtype=np.float32)
    
    # textural transform
    # img = cv2.GaussianBlur(img, (5, 5), random.uniform(0, sigma)) if sigma_ else img
    # img = colorJetting(img, eta) if eta_ else img
    # img = imgOcclusion(img, gamma) if gamma_ else img
    
    # imgToTensorBefore = copy.deepcopy(img)
    # saveImgWithLandmarksInGetItem(imgToTensorBefore, pts, 'test1'+str(time.time()))
    
    img = (img - 127.5)/128
    img = img.transpose((2, 0, 1))
    pts = pts.reshape(-1,)
    
    return img, pts

