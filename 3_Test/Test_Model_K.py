import cv2
from PIL import Image
import numpy as np
from os import listdir, mkdir
from os.path import isfile, join, isdir
from tqdm import tqdm
import glob
from tqdm import tqdm
import time


import sys
sys.path.insert(1, '../1_Train')
from utils import PSNR, _rgb2ycbcr
from demo import tetrahedral_interp_4x, pad_image, FourSimplexInterp



# USER PARAMS
UPSCALE = 4     # upscaling factor
SAMPLING_INTERVAL = 4        # N bit uniform sampling
LUT_PATH = "Model_K_x{}_{}bit_int8.npy".format(UPSCALE, SAMPLING_INTERVAL)    # Trained SR net params
TEST_DIR = './test/'      # Test images



# Load LUT
LUT = np.load(LUT_PATH).astype(np.float32).reshape(-1, UPSCALE*UPSCALE)  # N(=(2^SAMPLING_INTERVAL + 1)^4D), 16(=r*r)



# Test LR images
files_lr = glob.glob(TEST_DIR + '/LR_x{}/*.png'.format(UPSCALE))
files_lr.sort()

# Test GT images
files_gt = glob.glob(TEST_DIR + '/HR/*.png')
files_gt.sort()


psnrs = []
times =[]

if not isdir('./output_K_x{}_{}bit'.format(UPSCALE, SAMPLING_INTERVAL)):
    mkdir('./output_K_x{}_{}bit'.format(UPSCALE, SAMPLING_INTERVAL))

i=0

for ti, fn in enumerate(tqdm(files_gt)):

    start_time = time.time()

    # Load LR image
    img_lr = cv2.imread(files_lr[ti]).astype(np.float32)

    img_lr_YUV =cv2.cvtColor(img_lr, cv2.COLOR_BGR2YUV)
    channel_Y = img_lr_YUV[:,:,0:1]
    channel_V = img_lr_YUV[:,:,1:2]
    channel_U = img_lr_YUV[:,:,2:3]

    # np.savetxt(".//yuv_np//frame_data_yuv_y_gt.txt",img_lr_YUV[:,:,0],fmt='%.3d')
    # np.savetxt(".//yuv_np//frame_data_yuv_u_gt.txt",img_lr_YUV[:,:,1],fmt='%.3d')
    # np.savetxt(".//yuv_np//frame_data_yuv_v_gt.txt",img_lr_YUV[:,:,2],fmt='%.3d')

    # Load GT image
    img_gt = cv2.imread(files_gt[ti])
    

    # Sampling interval for input
    q = 2**SAMPLING_INTERVAL


    
    channel_sr_V =cv2.resize(channel_V, (0,0), fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_LINEAR)
    channel_sr_U =cv2.resize(channel_U, (0,0), fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_LINEAR)

    channel_sr_V =(channel_sr_V-140).astype(np.uint8)
    channel_sr_U =(channel_sr_U+190).astype(np.uint8)


    # out_r0 = tetrahedral_interp_4x(LUT, pad_image(channel_Y,2,2,2,2), img_lr.shape[0], img_lr.shape[1],UPSCALE, q, SAMPLING_INTERVAL)


    # out_r1 = tetrahedral_interp_4x(LUT, np.rot90(pad_image(channel_Y,2,2,2,2),1), img_lr.shape[1], img_lr.shape[0],UPSCALE, q, SAMPLING_INTERVAL)
    # out_r1 =np.rot90(out_r1, 3)

    # out_r2 = tetrahedral_interp_4x(LUT,np.rot90(pad_image(channel_Y,2,2,2,2),2), img_lr.shape[0], img_lr.shape[1],UPSCALE ,q, SAMPLING_INTERVAL)
    # out_r2 =np.rot90(out_r2, 2)

    # out_r3 = tetrahedral_interp_4x(LUT, np.rot90(pad_image(channel_Y,2,2,2,2),3), img_lr.shape[1], img_lr.shape[0],UPSCALE ,q, SAMPLING_INTERVAL)
    # out_r3 =np.rot90(out_r3, 1)

    # print(out_r0.shape,out_r1.shape,out_r2.shape,out_r3.shape)

    # channel_sr_Y = (out_r0/1.0 + out_r1/1.0 + out_r2/1.0 + out_r3/1.0)/255.0
    # channel_sr_Y =np.round(np.clip(channel_sr_Y, 0, 1) * 255).astype(np.uint8)

    



    # img_out =cv2.merge((channel_sr_Y, channel_sr_U, channel_sr_V))
    # img_out[:,:,1] =img_out[:,:,1]-127
    # np.savetxt(".//yuv_np//frame_data_yuv_y.txt",img_out[:,:,0],fmt='%.2d')
    # np.savetxt(".//yuv_np//frame_data_yuv_u.txt",img_out[:,:,1],fmt='%.2d')
    # np.savetxt(".//yuv_np//frame_data_yuv_v.txt",img_out[:,:,2],fmt='%.3d')
    # img_out = cv2.cvtColor(img_out, cv2.COLOR_YUV2RGB)
    # np.savetxt(".//rgb_np//frame_data_rgb_r.txt",img_out[:,:,0],fmt='%.2d')
    # np.savetxt(".//rgb_np//frame_data_rgb_g.txt",img_out[:,:,1],fmt='%.2d')
    # np.savetxt(".//rgb_np//frame_data_rgb_b.txt",img_out[:,:,2],fmt='%.2d')

    img_lr =channel_Y
    h = img_lr.shape[0]
    w = img_lr.shape[1]
    img_in = np.pad(img_lr, ((0,1), (0,1), (0,0)), mode='reflect').transpose((2,0,1))
    out_r0 = FourSimplexInterp(LUT, img_in, h, w, q, 0,4, upscale=UPSCALE)

    img_in = np.pad(np.rot90(img_lr, 1), ((0,1), (0,1), (0,0)), mode='reflect').transpose((2,0,1))
    out_r1 = FourSimplexInterp(LUT, img_in, w, h, q, 3,4, upscale=UPSCALE)

    img_in = np.pad(np.rot90(img_lr, 2), ((0,1), (0,1), (0,0)), mode='reflect').transpose((2,0,1))
    out_r2 = FourSimplexInterp(LUT, img_in, h, w, q, 2,4, upscale=UPSCALE)

    img_in = np.pad(np.rot90(img_lr, 3), ((0,1), (0,1), (0,0)), mode='reflect').transpose((2,0,1))
    out_r3 = FourSimplexInterp(LUT, img_in, w, h, q, 1,4, upscale=UPSCALE)

    img_out = (out_r0/1.0 + out_r1/1.0 + out_r2/1.0 + out_r3/1.0) / 255.0
    img_out = img_out.transpose((1,2,0))
    img_out = np.round(np.clip(img_out, 0, 1) * 255).astype(np.uint8)

    img_out =cv2.merge((img_out, channel_sr_U, channel_sr_V))
    img_out[:,:,1] =img_out[:,:,1].astype(np.int8)

    # np.savetxt(".//yuv_np//frame_data_yuv_y.txt",img_out[:,:,0],fmt='%.2d')
    # np.savetxt(".//yuv_np//frame_data_yuv_u.txt",img_out[:,:,1],fmt='%.2d')
    # np.savetxt(".//yuv_np//frame_data_yuv_v.txt",img_out[:,:,2],fmt='%.3d')

    img_out = cv2.cvtColor(img_out, cv2.COLOR_YUV2RGB)



    times.append(time.time() - start_time)

    # Matching image sizes 
    if img_gt.shape[0] < img_out.shape[0]:
        img_out = img_out[:img_gt.shape[0]]
    if img_gt.shape[1] < img_out.shape[1]:
        img_out = img_out[:, :img_gt.shape[1]]

    if img_gt.shape[0] > img_out.shape[0]:
        img_out = np.pad(img_out, ((0,img_gt.shape[0]-img_out.shape[0]),(0,0),(0,0)))
    if img_gt.shape[1] > img_out.shape[1]:
        img_out = np.pad(img_out, ((0,0),(0,img_gt.shape[1]-img_out.shape[1]),(0,0)))

    # Save to file
    Image.fromarray(img_out).save('./output_K_x{}_{}bit/{}_LUT_interp_{}bit.png'.format(UPSCALE, SAMPLING_INTERVAL, fn.split('/')[-1][:-4], SAMPLING_INTERVAL))

    CROP_S = 4
    psnr = PSNR(_rgb2ycbcr(img_gt)[:,:,0], _rgb2ycbcr(img_out)[:,:,0], CROP_S)
    psnrs.append(psnr)

print('AVG PSNR: {}'.format(np.mean(np.asarray(psnrs))))
print('AVG TIME: {}'.format(np.mean(np.asarray(times))))


