import tqdm
import copy
import torch
import numpy as np
import cv2
from models import Baseline, loadWeights, BaselineSepDyBN
from datasets import WFLW256, datasetSize, kptNum
import time
from ptflops import get_model_complexity_info

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def testWFLW256Baseline(mode, test_epoch, logName):
    testset = WFLW256(mode=mode, augment=False)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True)

    macs, params = get_model_complexity_info(Baseline(), (3, 128, 128), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    
    model = Baseline()
    model = loadWeights(model, './checkpoints/'+logName+'_model_'+str(test_epoch)+'.pth', 'cuda:0')
    model.eval().cuda('cuda:0')

    errorRates = []
    times = []
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader):
            img, tpts = data
            img = img.cuda('cuda:0').float()
            tpts = tpts.squeeze().numpy().reshape(-1, 2)

            startTime = time.time()
            pts = model(img)
            times.append(time.time()-startTime)
            pts = pts.cpu().squeeze().numpy().reshape(-1, 2)
            
            normalizeFactor = np.sqrt((tpts[60, 0] - tpts[72, 0])**2 + (tpts[60, 1] - tpts[72, 1])**2)
            errorRate = np.sum(np.sqrt(np.sum(pow(pts-tpts, 2), axis=1)))/kptNum/normalizeFactor
            errorRates.append(errorRate)

    errorRate = sum(errorRates) / datasetSize[mode] * 100
    print(mode + ' error rate: ' + str(errorRate))
    print("Avg forward time is: " + str(sum(times)/datasetSize[mode]) + "ms")
    print("FPS: " + str(1/sum(times)*datasetSize[mode]))

    return errorRate


def testWFLW256BaselineSepDyBN(mode, test_epoch, logName):
    testset = WFLW256(mode=mode, augment=False)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True)

    macs, params = get_model_complexity_info(BaselineSepDyBN(temp=1), (3, 128, 128), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    
    model = BaselineSepDyBN(temp=1)
    model = loadWeights(model, './checkpoints/'+logName+'_model_'+str(test_epoch)+'.pth', 'cuda:0')
    model.eval().cuda('cuda:0')

    errorRates = []
    times = []
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader):
            img, tpts = data
            img = img.cuda('cuda:0').float()
            tpts = tpts.squeeze().numpy().reshape(-1, 2)

            startTime = time.time()
            pts = model(img)
            times.append(time.time()-startTime)
            pts = pts.cpu().squeeze().numpy().reshape(-1, 2)
            
            normalizeFactor = np.sqrt((tpts[60, 0] - tpts[72, 0])**2 + (tpts[60, 1] - tpts[72, 1])**2)
            errorRate = np.sum(np.sqrt(np.sum(pow(pts-tpts, 2), axis=1)))/kptNum/normalizeFactor
            errorRates.append(errorRate)

    errorRate = sum(errorRates) / datasetSize[mode] * 100
    print(mode + ' error rate: ' + str(errorRate))
    print("Avg forward time is: " + str(sum(times)/datasetSize[mode]) + "ms")
    print("FPS: " + str(1/sum(times)*datasetSize[mode]))

    return errorRate


if __name__ == "__main__":
    testWFLW256Baseline('test', 'final', '20210101_WFLW256Baseline_Train_1')
    testWFLW256BaselineSepDyBN('test', 'final', '20210101_WFLW256BaselineSepDyBN_Train_1')
