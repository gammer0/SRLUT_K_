import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from   torch.autograd import Variable

import cv2
from PIL import Image
import numpy as np
import time
from os import mkdir
from os.path import join,isdir
from tqdm import tqdm
import glob

from utils import PSNR, GeneratorEnqueuer, DirectoryIterator_DIV2K, _load_img_array, _rgb2ycbcr
from tensorboardX import SummaryWriter


EXP_NAME = "SR-LUT"
VERSION = "K"
UPSCALE = 4     # upscaling factor

NB_BATCH = 32        # mini-batch
CROP_SIZE = 48       # input LR training patch size

START_ITER = 100000      # Set 0 for from scratch, else will load saved params and trains further
NB_ITER = 102000    # Total number of training iterations

I_DISPLAY = 100     # display info every N iteration
I_VALIDATION = 1000  # validate every N iteration
I_SAVE = 1000       # save models every N iteration

TRAIN_DIR = './train/'  # Training images: png files should just locate in the directory (eg ./train/img0001.png ... ./train/img0800.png)
VAL_DIR = './valid/'      # Validation images

LR_G = 1e-4

writer = SummaryWriter(log_dir='./log/{}'.format(str(VERSION)))


class SRNet(torch.nn.Module):
    def __init__(self, upscale=4):
        super(SRNet, self).__init__()

        self.upscale = upscale

        self.conv1 = nn.Conv2d(1, 64, [2,2], stride=1, padding=0, dilation=1)
        self.conv2 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv3 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv4 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv5 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv6 = nn.Conv2d(64, 1*upscale*upscale, 1, stride=1, padding=0, dilation=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale)

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.kaiming_normal(m.weight)
                nn.init.constant(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


    def forward(self, x_in):
        B, C, H, W = x_in.size()
        x_in = x_in.reshape(B*C, 1, H, W) #单通道无需重塑

        x = self.conv1(x_in)
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))
        x = self.conv4(F.relu(x))
        x = self.conv5(F.relu(x))
        x = self.conv6(F.relu(x))
        x = self.pixel_shuffle(x)
        x = x.reshape(B, C, self.upscale*(H-1), self.upscale*(W-1))

        return x



def main():

    model_G =SRNet(upscale=UPSCALE).cuda()

    params_G =list(filter(lambda p: p.requires_grad, model_G.parameters()))
    opt_G = optim.Adam(params_G, lr=LR_G)

    if START_ITER > 0:
        lm = torch.load('checkpoint/{}/model_G_i{:06d}.pth'.format(str(VERSION), START_ITER))
        model_G.load_state_dict(lm.state_dict(), strict=True)

        lm = torch.load('checkpoint/{}/opt_G_i{:06d}.pth'.format(str(VERSION), START_ITER))
        opt_G.load_state_dict(lm.state_dict())

    Iter_H = GeneratorEnqueuer(DirectoryIterator_DIV2K( 
                                datadir = TRAIN_DIR,
                                crop_size = CROP_SIZE, 
                                crop_per_image = NB_BATCH//4,
                                out_batch_size = NB_BATCH,
                                scale_factor = UPSCALE,
                                shuffle=True))
    Iter_H.start(max_q_size=16,workers=12)

    if not isdir('checkpoint'):
        mkdir('checkpoint')
    if not isdir('result'):
        mkdir('result')
    if not isdir('checkpoint/{}'.format(str(VERSION))):
        mkdir('checkpoint/{}'.format(str(VERSION)))
    if not isdir('result/{}'.format(str(VERSION))):
        mkdir('result/{}'.format(str(VERSION)))
    if not isdir('result/{}/HR'.format(str(VERSION))):
        mkdir('result/{}/HR'.format(str(VERSION)))

    print('===> Training start')
    l_accum = [0.,0.,0.]
    dT = 0.
    rT = 0.
    accum_samples = 0


    def SaveCheckpoint(i, best=False):
        str_best = ''
        if best:
            str_best = '_best'

        torch.save(model_G, 'checkpoint/{}/model_G_i{:06d}{}.pth'.format(str(VERSION), i, str_best ))
        torch.save(opt_G, 'checkpoint/{}/opt_G_i{:06d}{}.pth'.format(str(VERSION), i, str_best))
        print("Checkpoint saved")



    for i in tqdm(range(START_ITER+1, NB_ITER+1)):

        model_G.train()

        # Data preparing
        st = time.time()
        batch_L, batch_H = Iter_H.dequeue()
        batch_H = Variable(torch.from_numpy(batch_H)).cuda()      # BxCxHxW, range [0,1]
        batch_L = Variable(torch.from_numpy(batch_L)).cuda()      # BxCxHxW, range [0,1]
        dT += time.time() - st


        ## TRAIN G
        st = time.time()
        opt_G.zero_grad()

        # Rotational ensemble training
        batch_S1 = model_G(F.pad(batch_L, (0,1,0,1), mode='reflect'))

        batch_S2 = model_G(F.pad(torch.rot90(batch_L, 1, [2,3]), (0,1,0,1), mode='reflect'))
        batch_S2 = torch.rot90(batch_S2, 3, [2,3])

        batch_S3 = model_G(F.pad(torch.rot90(batch_L, 2, [2,3]), (0,1,0,1), mode='reflect'))
        batch_S3 = torch.rot90(batch_S3, 2, [2,3])

        batch_S4 = model_G(F.pad(torch.rot90(batch_L, 3, [2,3]), (0,1,0,1), mode='reflect'))
        batch_S4 = torch.rot90(batch_S4, 1, [2,3])


        batch_S = ( torch.clamp(batch_S1,-1,1)*127 + torch.clamp(batch_S2,-1,1)*127 )
        batch_S += ( torch.clamp(batch_S3,-1,1)*127 + torch.clamp(batch_S4,-1,1)*127 )
        batch_S /= 255.0

        loss_Pixel = torch.mean( ((batch_S - batch_H)**2)  )
        loss_G = loss_Pixel

        # Update
        loss_G.backward()
        opt_G.step()
        rT += time.time() - st

        # For monitoring
        accum_samples += NB_BATCH
        l_accum[0] += loss_Pixel.item()


        ## Show information
        if i % I_DISPLAY == 0:
            writer.add_scalar('loss_Pixel', l_accum[0]/I_DISPLAY, i)

            print("{} {}| Iter:{:6d}, Sample:{:6d}, GPixel:{:.2e}, dT:{:.4f}, rT:{:.4f}".format(
                EXP_NAME, VERSION, i, accum_samples, l_accum[0]/I_DISPLAY, dT/I_DISPLAY, rT/I_DISPLAY))
            l_accum = [0.,0.,0.]
            dT = 0.
            rT = 0.


        ## Save models
        if i % I_SAVE == 0:
            SaveCheckpoint(i)


        ## Validation
        if i % I_VALIDATION == 0:
            with torch.no_grad():
                model_G.eval()

                # Test for validation images
                files_gt = glob.glob(VAL_DIR + '/HR/*.png')
                files_gt.sort()
                files_lr = glob.glob(VAL_DIR + '/LR/*.png')
                files_lr.sort()

                psnrs = []
                

                for ti, fn in enumerate(files_gt):
                    # Load HR image
                    tmp = _load_img_array(files_gt[ti])
                    val_H = np.asarray(tmp).astype(np.float32)  # HxWxC



                    # Load LR image
                    tmp = _load_img_array(files_lr[ti])

                    tmp =cv2.cvtColor(tmp, cv2.COLOR_RGB2YUV)

                    val_L =tmp[:,:,0:1]

                    val_L = np.asarray(tmp).astype(np.float32)  # HxWxC
                    val_L = np.transpose(val_L, [2, 0, 1])      # CxHxW
                    val_L = val_L[np.newaxis, ...]            # BxCxHxW

                    val_L = Variable(torch.from_numpy(val_L.copy()), volatile=True).cuda()
                        
                    # Run model
                    batch_S1 = model_G(F.pad(val_L, (0,1,0,1), mode='reflect'))

                    batch_S2 = model_G(F.pad(torch.rot90(val_L, 1, [2,3]), (0,1,0,1), mode='reflect'))
                    batch_S2 = torch.rot90(batch_S2, 3, [2,3])

                    batch_S3 = model_G(F.pad(torch.rot90(val_L, 2, [2,3]), (0,1,0,1), mode='reflect'))
                    batch_S3 = torch.rot90(batch_S3, 2, [2,3])

                    batch_S4 = model_G(F.pad(torch.rot90(val_L, 3, [2,3]), (0,1,0,1), mode='reflect'))
                    batch_S4 = torch.rot90(batch_S4, 1, [2,3])
                
                    batch_S = ( torch.clamp(batch_S1,-1,1)*127 + torch.clamp(batch_S2,-1,1)*127 )
                    batch_S += ( torch.clamp(batch_S3,-1,1)*127 + torch.clamp(batch_S4,-1,1)*127 )
                    batch_S /= 255.0

                    val_L_U =tmp[:,:,1:2]
                    val_L_V =tmp[:,:,2:3]

                    image_sr_U =cv2.resize(val_L_U, (val_H.shape[1],val_H.shape[0]), interpolation=cv2.INTER_LINEAR)
                    image_sr_V =cv2.resize(val_L_V, (val_H.shape[1],val_H.shape[0]), interpolation=cv2.INTER_LINEAR)


                    # Output 
                    image_sr_Y = (batch_S).cpu().data.numpy()
                    image_sr_Y = np.clip(image_sr_Y[0], 0. , 1.)      # CxHxW
                    image_sr_Y = np.transpose(image_sr_Y, [1, 2, 0])  # HxWxC

                    # Save to file
                    image_sr_Y = ((image_sr_Y)*255).astype(np.uint8)
                    image_sr_Y = np.sum(image_sr_Y * [0.299, 0.587, 0.114], axis=2).astype(np.uint8)
                    image_sr_U = (image_sr_U+190).astype(np.uint8)
                    image_sr_V = (image_sr_V-140).astype(np.uint8)


                    image_out =cv2.merge((image_sr_Y,image_sr_U,image_sr_V))
                    image_out =cv2.cvtColor(image_out, cv2.COLOR_YUV2RGB)

                    Image.fromarray(image_out).save('result/{}/{}.png'.format(str(VERSION), fn.split('/')[-1]))

                    # PSNR on Y channel
                    img_gt = (val_H*255).astype(np.uint8)
                    CROP_S = 4
                    psnrs.append(PSNR(_rgb2ycbcr(img_gt)[:,:,0], _rgb2ycbcr(image_out)[:,:,0], CROP_S))

            print('AVG PSNR: Validation: {}'.format(np.mean(np.asarray(psnrs))))

            writer.add_scalar('PSNR_valid', np.mean(np.asarray(psnrs)), i)
            writer.flush()

if __name__ == '__main__':

    main()
