import os
import time
import math
import tqdm
import torch
import numpy as np
import logging
from torch.autograd import Variable

from models import BaselineSepDyBN, dataPrefetcher
from datasets import WFLW256, datasetSize


if not os.path.exists('/media/WDC/savedModels/WFLWSepBN'):
    os.mkdir('/media/WDC/savedModels/WFLWSepBN')
if not os.path.exists('./logs'):
    os.mkdir('./logs')

logName = '20210101_WFLW256BaselineSepDyBN_Train_1'
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler('./logs/' + logName + '.txt')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s  %(levelname)s  %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info(logName)

torch.backends.cudnn.benchmark = True

logger.info('Network: BaselineSepDyBN()')
logger.info('GPU: \'cuda:0\'')
model = BaselineSepDyBN()
model = model.cuda('cuda:0')
model.train()

logger.info('SGD：lr=0.001, weight_decay=5e-4, momentum=0.9')
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=5e-4, momentum=0.9)

logger.info('Training epoch: 2000')
logger.info('Lr schedule: cosine, warm epoch: 800, base max lr: 1e-3, base min lr: 1e-10')
logger.info('Dataset: WFLW256(mode="train", augment=True)')
logger.info('Dataloader: batch_size=8, shuffle=True, num_workers=8, pin_memory=True, drop_last=True')
trainSet = WFLW256(mode="train", augment=True)
maxEpoch = 2000
batchSize = 8
numWorkers = 4
itersPerBatch = datasetSize["train"]//batchSize

warmEpoch = 800
epochSize = itersPerBatch
lrMax = 1e-3*np.sqrt(batchSize//8)
lrMin = 1e-10*np.sqrt(batchSize//8)
iteration = 0

errorRate=100.0

for epoch in range(0, maxEpoch):
    print("Epoch: " + str(epoch))
    st = time.time()
    lossRecordPerEpoch = []

    dataLoader = torch.utils.data.DataLoader(trainSet, batch_size=batchSize, shuffle=True, num_workers=numWorkers, pin_memory=True, drop_last=True)
    prefetcher = dataPrefetcher(iter(dataLoader))

    for _ in tqdm.tqdm(range(itersPerBatch)):

        img, tpts = prefetcher.next()
        assert img.size()[0] == tpts.size()[0] == batchSize

        img = Variable(img.cuda('cuda:0'))
        tpts = Variable(tpts.cuda('cuda:0'))

        if iteration <= epochSize*warmEpoch:
            optimizer.param_groups[0]['lr'] = lrMin + (lrMax - lrMin)*iteration/(epochSize*warmEpoch)
        else:
            t1 = iteration - warmEpoch * epochSize
            t2 = (maxEpoch - warmEpoch) * epochSize
            t = t1 * math.pi / t2
            optimizer.param_groups[0]['lr'] = lrMin + (lrMax - lrMin) * 0.5 * (1 + math.cos(t))

        optimizer.zero_grad()
        out = model(img)
        loss = model.calculateLoss(out, tpts)
        loss.backward()
        optimizer.step()

        lossRecordPerEpoch.append(loss.item())

        iteration += 1

    epochTime = time.time() - st
    logger.info('epoch: ' + str(epoch) + ' lr: ' + str(optimizer.param_groups[0]['lr']) + ' loss: ' + str(sum(lossRecordPerEpoch)/len(lossRecordPerEpoch)) + ' time: ' + str(epochTime))

    torch.save(model.state_dict(), '/media/WDC/savedModels/WFLWSepBN/' + logName + '_model_'+str(epoch)+'.pth')

    model.updata_temperature()

torch.save(model.state_dict(), '/media/WDC/savedModels/WFLWSepBN/' + logName + '_model_final.pth')
logger.info("Training exit safely.")
