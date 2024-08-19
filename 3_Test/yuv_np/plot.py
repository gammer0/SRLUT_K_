import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    
    pixel_distribution("y")
    pixel_distribution("v")
    pixel_distribution("u")


def pixel_distribution(filename):
    data_yuv_u_gt = np.loadtxt('frame_data_yuv_{}_gt.txt'.format(filename))
    data_yuv_u =np.loadtxt('frame_data_yuv_{}.txt'.format(filename))

    data_series_u =pd.Series(data_yuv_u.flatten())
    data_series_u_gt =pd.Series(data_yuv_u_gt.flatten())

    data_series_u_values = data_series_u.value_counts()
    data_series_u_gt_values = data_series_u_gt.value_counts()

    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.bar(data_series_u_values.index, data_series_u_values.values)
    plt.xlim(-255,255)
    plt.xticks(np.arange(-255,255,50))
    plt.title('data_yuv_{}'.format(filename))
    plt.text(0,0,"O")
    plt.subplot(122)
    plt.bar(data_series_u_gt_values.index, data_series_u_gt_values.values)
    plt.xlim(-255,255)
    plt.xticks(np.arange(-255,255,50))
    plt.title('data_yuv_{}_gt'.format(filename))
    plt.text(0,0,"O")
    plt.savefig('pixel_distribution_{}.png'.format(filename))



if __name__ == '__main__':
    main() 
