import os
import cv2
import time
import numpy as np

def tetrahedral_interp_4x(weight, img_in, height, width, upscale,q,interval):
    img_out =np.zeros((height*upscale, width*upscale), dtype=np.float32)
    L = 2**(8-interval) + 1
    ll =L*L

    for y in range(2, img_in.shape[0] - 2):
        for x in range(2, img_in.shape[1] - 2):
            base = img_in[y, x].astype(np.int_) * ll
            ir = base + img_in[y, x + 1].astype(np.int_) * L + img_in[y, x + 2].astype(np.int_)
            it = base + img_in[y - 1, x].astype(np.int_) * L + img_in[y - 2, x].astype(np.int_)
            il = base + img_in[y, x - 1].astype(np.int_) * L + img_in[y, x - 2].astype(np.int_)
            ib = base + img_in[y + 1, x].astype(np.int_) * L + img_in[y + 2, x].astype(np.int_)

            img_out[2 * (y - 2), 2 * (x - 2)] = weight[ir][0] + weight[it][1] + weight[il][3] + weight[ib][2]
            img_out[2 * (y - 2), 2 * (x - 2) + 1] = weight[ir][1] + weight[it][3] + weight[il][2] + weight[ib][0]
            img_out[2 * (y - 2) + 1, 2 * (x - 2)] = weight[ir][2] + weight[it][0] + weight[il][1] + weight[ib][3]
            img_out[2 * (y - 2) + 1, 2 * (x - 2) + 1] = weight[ir][3] + weight[it][2] + weight[il][0] + weight[ib][1]

    return img_out/q
def pad_image(image, top, bottom, left, right): #可能有问题
    """对图像进行边缘填充"""
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT)
def Frame_SR(lut, upscale, path):
    path = path +"//lq"
    files_lr =[]

    for file in os.listdir(path):
        if file.endswith(".png"):
            files_lr.append(str(file))
    
    files_lr.sort(key=lambda x: int(x[:4]))
    
    print(f"begin, image num ={len(files_lr)}")
    TIME = []

    for file in files_lr:
        img_lr = cv2.imread(path+"//"+file)
        
        #时间记录
        start_time = time.time()

        img_lr_YUV =cv2.cvtColor(img_lr, cv2.COLOR_BGR2YUV)
        channel_Y = img_lr_YUV[:,:,0]
        channel_V = img_lr_YUV[:,:,1]
        channel_U = img_lr_YUV[:,:,2]

        channel_sr_V =cv2.resize(channel_V, (0,0), fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        channel_sr_U =cv2.resize(channel_U, (0,0), fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)

        channel_sr_Y = tetrahedral_interp_4x(lut,img_in=pad_image(channel_Y, 2, 2, 2, 2),height=img_lr.shape[0],width=img_lr.shape[1],upscale=4) 

        # 时间记录
        end_time = time.time()
        TIME.append(end_time-start_time)

        img_sr_YUV = cv2.merge([channel_sr_Y, channel_sr_U, channel_sr_V])
        img_sr = cv2.cvtColor(img_sr_YUV, cv2.COLOR_YUV2BGR)
        
        if not os.path.exists(path+"//re"):
            os.makedirs(path+"//re")

        cv2.imwrite(path+"//re//"+file, img_sr)





def FourSimplexInterp(weight, img_in, h, w, q, rot, SAMPLING_INTERVAL, upscale=4):
    L = 2**(8-SAMPLING_INTERVAL) + 1

    # Extract MSBs
    img_a1 = img_in[:, 0:0+h, 0:0+w] // q
    img_b1 = img_in[:, 0:0+h, 1:1+w] // q
    img_c1 = img_in[:, 1:1+h, 0:0+w] // q
    img_d1 = img_in[:, 1:1+h, 1:1+w] // q
        
    img_a2 = img_a1 + 1
    img_b2 = img_b1 + 1
    img_c2 = img_c1 + 1
    img_d2 = img_d1 + 1

    # Extract LSBs
    fa_ = img_in[:, 0:0+h, 0:0+w] % q
    fb_ = img_in[:, 0:0+h, 1:1+w] % q
    fc_ = img_in[:, 1:1+h, 0:0+w] % q
    fd_ = img_in[:, 1:1+h, 1:1+w] % q


    # Vertices (O in Eq3 and Table3 in the paper)
    p0000 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0001 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0010 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0011 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0100 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0101 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0110 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0111 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
        
    p1000 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1001 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1010 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1011 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1100 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1101 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1110 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1111 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
        
    # Output image holder
    out = np.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))

    # Naive pixelwise output value interpolation (Table3 in the paper)
    # It would be faster implemented with a parallel operation
    for c in range(img_a1.shape[0]):
        for y in range(img_a1.shape[1]):
            for x in range(img_a1.shape[2]):
                fa = fa_[c,y,x]
                fb = fb_[c,y,x]
                fc = fc_[c,y,x]
                fd = fd_[c,y,x]

                if fa > fb:
                    if fb > fc:
                        if fc > fd:
                            out[c,y,x] = (q-fa) * p0000[c,y,x] + (fa-fb) * p1000[c,y,x] + (fb-fc) * p1100[c,y,x] + (fc-fd) * p1110[c,y,x] + (fd) * p1111[c,y,x]
                        elif fb > fd:
                            out[c,y,x] = (q-fa) * p0000[c,y,x] + (fa-fb) * p1000[c,y,x] + (fb-fd) * p1100[c,y,x] + (fd-fc) * p1101[c,y,x] + (fc) * p1111[c,y,x]
                        elif fa > fd:
                            out[c,y,x] = (q-fa) * p0000[c,y,x] + (fa-fd) * p1000[c,y,x] + (fd-fb) * p1001[c,y,x] + (fb-fc) * p1101[c,y,x] + (fc) * p1111[c,y,x]
                        else:
                            out[c,y,x] = (q-fd) * p0000[c,y,x] + (fd-fa) * p0001[c,y,x] + (fa-fb) * p1001[c,y,x] + (fb-fc) * p1101[c,y,x] + (fc) * p1111[c,y,x]
                    elif fa > fc:
                        if fb > fd:
                            out[c,y,x] = (q-fa) * p0000[c,y,x] + (fa-fc) * p1000[c,y,x] + (fc-fb) * p1010[c,y,x] + (fb-fd) * p1110[c,y,x] + (fd) * p1111[c,y,x]
                        elif fc > fd:
                            out[c,y,x] = (q-fa) * p0000[c,y,x] + (fa-fc) * p1000[c,y,x] + (fc-fd) * p1010[c,y,x] + (fd-fb) * p1011[c,y,x] + (fb) * p1111[c,y,x]
                        elif fa > fd:
                            out[c,y,x] = (q-fa) * p0000[c,y,x] + (fa-fd) * p1000[c,y,x] + (fd-fc) * p1001[c,y,x] + (fc-fb) * p1011[c,y,x] + (fb) * p1111[c,y,x]
                        else:
                            out[c,y,x] = (q-fd) * p0000[c,y,x] + (fd-fa) * p0001[c,y,x] + (fa-fc) * p1001[c,y,x] + (fc-fb) * p1011[c,y,x] + (fb) * p1111[c,y,x]
                    else:
                        if fb > fd:
                            out[c,y,x] = (q-fc) * p0000[c,y,x] + (fc-fa) * p0010[c,y,x] + (fa-fb) * p1010[c,y,x] + (fb-fd) * p1110[c,y,x] + (fd) * p1111[c,y,x]
                        elif fc > fd:
                            out[c,y,x] = (q-fc) * p0000[c,y,x] + (fc-fa) * p0010[c,y,x] + (fa-fd) * p1010[c,y,x] + (fd-fb) * p1011[c,y,x] + (fb) * p1111[c,y,x]
                        elif fa > fd:
                            out[c,y,x] = (q-fc) * p0000[c,y,x] + (fc-fd) * p0010[c,y,x] + (fd-fa) * p0011[c,y,x] + (fa-fb) * p1011[c,y,x] + (fb) * p1111[c,y,x]
                        else:
                            out[c,y,x] = (q-fd) * p0000[c,y,x] + (fd-fc) * p0001[c,y,x] + (fc-fa) * p0011[c,y,x] + (fa-fb) * p1011[c,y,x] + (fb) * p1111[c,y,x]

                else:
                    if fa > fc:
                        if fc > fd:
                            out[c,y,x] = (q-fb) * p0000[c,y,x] + (fb-fa) * p0100[c,y,x] + (fa-fc) * p1100[c,y,x] + (fc-fd) * p1110[c,y,x] + (fd) * p1111[c,y,x]
                        elif fa > fd:
                            out[c,y,x] = (q-fb) * p0000[c,y,x] + (fb-fa) * p0100[c,y,x] + (fa-fd) * p1100[c,y,x] + (fd-fc) * p1101[c,y,x] + (fc) * p1111[c,y,x]
                        elif fb > fd:
                            out[c,y,x] = (q-fb) * p0000[c,y,x] + (fb-fd) * p0100[c,y,x] + (fd-fa) * p0101[c,y,x] + (fa-fc) * p1101[c,y,x] + (fc) * p1111[c,y,x]
                        else:
                            out[c,y,x] = (q-fd) * p0000[c,y,x] + (fd-fb) * p0001[c,y,x] + (fb-fa) * p0101[c,y,x] + (fa-fc) * p1101[c,y,x] + (fc) * p1111[c,y,x]
                    elif fb > fc:
                        if fa > fd:
                            out[c,y,x] = (q-fb) * p0000[c,y,x] + (fb-fc) * p0100[c,y,x] + (fc-fa) * p0110[c,y,x] + (fa-fd) * p1110[c,y,x] + (fd) * p1111[c,y,x]
                        elif fc > fd:
                            out[c,y,x] = (q-fb) * p0000[c,y,x] + (fb-fc) * p0100[c,y,x] + (fc-fd) * p0110[c,y,x] + (fd-fa) * p0111[c,y,x] + (fa) * p1111[c,y,x]
                        elif fb > fd:
                            out[c,y,x] = (q-fb) * p0000[c,y,x] + (fb-fd) * p0100[c,y,x] + (fd-fc) * p0101[c,y,x] + (fc-fa) * p0111[c,y,x] + (fa) * p1111[c,y,x]
                        else:
                            out[c,y,x] = (q-fd) * p0000[c,y,x] + (fd-fb) * p0001[c,y,x] + (fb-fc) * p0101[c,y,x] + (fc-fa) * p0111[c,y,x] + (fa) * p1111[c,y,x]
                    else:
                        if fa > fd:
                            out[c,y,x] = (q-fc) * p0000[c,y,x] + (fc-fb) * p0010[c,y,x] + (fb-fa) * p0110[c,y,x] + (fa-fd) * p1110[c,y,x] + (fd) * p1111[c,y,x]
                        elif fb > fd:
                            out[c,y,x] = (q-fc) * p0000[c,y,x] + (fc-fb) * p0010[c,y,x] + (fb-fd) * p0110[c,y,x] + (fd-fa) * p0111[c,y,x] + (fa) * p1111[c,y,x]
                        elif fc > fd:
                            out[c,y,x] = (q-fc) * p0000[c,y,x] + (fc-fd) * p0010[c,y,x] + (fd-fb) * p0011[c,y,x] + (fb-fa) * p0111[c,y,x] + (fa) * p1111[c,y,x]
                        else:
                            out[c,y,x] = (q-fd) * p0000[c,y,x] + (fd-fc) * p0001[c,y,x] + (fc-fb) * p0011[c,y,x] + (fb-fa) * p0111[c,y,x] + (fa) * p1111[c,y,x]

    out = np.transpose(out, (0, 1,3, 2,4)).reshape((img_a1.shape[0], img_a1.shape[1]*upscale, img_a1.shape[2]*upscale))
    out = np.rot90(out, rot, [1,2])
    out = out / q
    return out
