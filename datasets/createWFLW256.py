import cv2
import copy
import tqdm
import numpy as np

import time

split = 'test_occlusion'  # train, test, test_largepose, test_illumination, test_expression ...
size = 256

if len(split.split('_')) == 1:
    with open('/media/WDC/datasets/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_'+split+'.txt') as f, open('/media/WDC/datasets/WFLW/WFLW256/WFLW256_'+split+'.txt', 'w') as fw:
        for idx, line in tqdm.tqdm(enumerate(f.readlines())):
            line = line.strip().split()
            landmarks = (np.array([float(x) for x in line[:-11]])).reshape(-1, 2)
            bbox = np.array([int(x) for x in line[-11:-7]])
            imgPath = '/media/WDC/datasets/WFLW/WFLW_images/'+line[-1]
            img = cv2.imread(imgPath)
            imgRawHeight, imgRawWidth = img.shape[0], img.shape[1]

            left, top, right, bottom = bbox
            left, right, top, bottom = int(left), int(right), int(top), int(bottom)
            
            width = right-left+1
            height = bottom-top+1

            longSize, shortSize = max(width, height), min(width, height)
            diff = longSize - shortSize
            correctDistance = diff//2
            if width > height:
                top = top - correctDistance
                bottom = bottom + correctDistance + diff%2
            else:
                left = left - correctDistance
                right = right + correctDistance + diff%2
            assert right - left == bottom - top
            if (right - left + 1)%2 == 1:
                right += 1
                bottom += 1  # right and bottom pixels are included in the crop img

            bboxSize = right-left+1
            assert bboxSize%2 == 0
            bboxSize2x = int(2*bboxSize)
            imgNew = np.zeros((bboxSize2x, bboxSize2x, 3))

            leftNew, rightNew, topNew, bottomNew = left-bboxSize//2, right+bboxSize//2, top-bboxSize//2, bottom+bboxSize//2
            leftLimited = leftNew if leftNew >= 0 else 0
            rightLimited = rightNew if rightNew < imgRawWidth else imgRawWidth - 1
            topLimited = topNew if topNew >= 0 else 0
            bottomLimited = bottomNew if bottomNew < imgRawHeight else imgRawHeight - 1
            
            imgNew[int(topLimited-topNew):int(bottomLimited-topNew+1), int(leftLimited-leftNew):int(rightLimited-leftNew+1), :] = \
                img[int(topLimited):int(bottomLimited+1), int(leftLimited):int(rightLimited+1), :]
            imgNew = np.array(imgNew, dtype=np.uint8)

            landmarks[:, 0] = landmarks[:, 0] - leftNew
            landmarks[:, 1] = landmarks[:, 1] - topNew

            imgNew = cv2.resize(imgNew, (size, size), interpolation=cv2.INTER_NEAREST)
            landmarks = landmarks*size/bboxSize2x

            cv2.imwrite('/media/WDC/datasets/WFLW/WFLW256/'+split+'/WFLW'+str(idx).zfill(4)+'.png', imgNew)

            # for i in range(98):
            #     cv2.circle(imgNew, (int(landmarks[i, 0]), int(landmarks[i, 1])), 2, (0,0,255), -1)
            # cv2.imwrite('./test'+str(time.time())+'.png', imgNew)

            for i in range(98):
                fw.write(str(landmarks[i, 0]) + ' ')  # x1, x2, ..., xN 
            for i in range(98):
                fw.write(str(landmarks[i, 1]) + ' ')  # y1, y2, ..., yN
            fw.write('/media/WDC/datasets/WFLW/WFLW256/'+split+'/WFLW'+str(idx).zfill(4)+'.png\n')  # absolute path

            # break

else:
    imgCatogory = split.split('_')[1]
    with open('/media/WDC/datasets/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt') as f, open('/media/WDC/datasets/WFLW/WFLW256/WFLW256_'+imgCatogory+'.txt', 'w') as fw:
        for idx, line in tqdm.tqdm(enumerate(f.readlines())):
            line = line.strip().split()
            category = [int(x) for x in line[-7:-1]]
            assert imgCatogory in ['largepose', 'expression', 'illumination', 'makeup', 'occlusion', 'blur']
            if imgCatogory == 'largepose' and category[0] != 1:
                continue
            elif imgCatogory == 'expression' and category[1] != 1:
                continue
            elif imgCatogory == 'illumination' and category[2] != 1:
                continue
            elif imgCatogory == 'makeup' and category[3] != 1:
                continue
            elif imgCatogory == 'occlusion' and category[4] != 1:
                continue
            elif imgCatogory == 'blur' and category[5] != 1:
                continue

            landmarks = (np.array([float(x) for x in line[:-11]])).reshape(-1, 2)
            bbox = np.array([int(x) for x in line[-11:-7]])
            imgPath = '/media/WDC/datasets/WFLW/WFLW_images/'+line[-1]
            img = cv2.imread(imgPath)
            imgRawHeight, imgRawWidth = img.shape[0], img.shape[1]

            left, top, right, bottom = bbox
            left, right, top, bottom = int(left), int(right), int(top), int(bottom)
            
            width = right-left+1
            height = bottom-top+1

            longSize, shortSize = max(width, height), min(width, height)
            diff = longSize - shortSize
            correctDistance = diff//2
            if width > height:
                top = top - correctDistance
                bottom = bottom + correctDistance + diff%2
            else:
                left = left - correctDistance
                right = right + correctDistance + diff%2
            assert right - left == bottom - top
            if (right - left + 1)%2 == 1:
                right += 1
                bottom += 1  # right and bottom pixels are included in the crop img

            bboxSize = right-left+1
            assert bboxSize%2 == 0
            bboxSize2x = int(2*bboxSize)
            imgNew = np.zeros((bboxSize2x, bboxSize2x, 3))

            leftNew, rightNew, topNew, bottomNew = left-bboxSize//2, right+bboxSize//2, top-bboxSize//2, bottom+bboxSize//2
            leftLimited = leftNew if leftNew >= 0 else 0
            rightLimited = rightNew if rightNew < imgRawWidth else imgRawWidth - 1
            topLimited = topNew if topNew >= 0 else 0
            bottomLimited = bottomNew if bottomNew < imgRawHeight else imgRawHeight - 1
            
            imgNew[int(topLimited-topNew):int(bottomLimited-topNew+1), int(leftLimited-leftNew):int(rightLimited-leftNew+1), :] = \
                img[int(topLimited):int(bottomLimited+1), int(leftLimited):int(rightLimited+1), :]
            imgNew = np.array(imgNew, dtype=np.uint8)

            landmarks[:, 0] = landmarks[:, 0] - leftNew
            landmarks[:, 1] = landmarks[:, 1] - topNew

            imgNew = cv2.resize(imgNew, (size, size), interpolation=cv2.INTER_NEAREST)
            landmarks = landmarks*size/bboxSize2x

            cv2.imwrite('/media/WDC/datasets/WFLW/WFLW256/'+imgCatogory+'/WFLW'+str(idx).zfill(4)+'.png', imgNew)

            # for i in range(98):
            #     cv2.circle(imgNew, (int(landmarks[i, 0]), int(landmarks[i, 1])), 2, (0,0,255), -1)
            # cv2.imwrite('./test'+str(time.time())+'.png', imgNew)

            for i in range(98):
                fw.write(str(landmarks[i, 0]) + ' ')  # x1, x2, ..., xN 
            for i in range(98):
                fw.write(str(landmarks[i, 1]) + ' ')  # y1, y2, ..., yN
            fw.write('/media/WDC/datasets/WFLW/WFLW256/'+imgCatogory+'/WFLW'+str(idx).zfill(4)+'.png\n')  # absolute path
