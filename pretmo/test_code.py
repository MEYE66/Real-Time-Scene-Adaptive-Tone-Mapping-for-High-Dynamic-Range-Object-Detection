import os
import numpy as np
import time
import cv2
import sys
import numpy.lib.format
import struct


BIT24 = 2**24

def fastload(file):
    if type(file) == str:
        file=open(file,"rb")
    header = file.read(128)
    if not header:
        return None
    descr = str(header[19:25], 'utf-8').replace("'","").replace(" ","")
    shape = tuple(int(num) for num in str(header[60:120], 'utf-8').replace(', }', '').replace('(', '').replace(')', '').split(','))
    datasize = numpy.lib.format.descr_to_dtype(descr).itemsize
    for dimension in shape:
        datasize *= dimension
    return np.ndarray(shape, dtype=descr, buffer=file.read(datasize))


if __name__ == '__main__':
    
    path = "/home/ligongzhe/mmdetection/test.tiff"
    
    data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    print(data.min(), data.max())
    
    # data = data / (BIT24-1)
    
    
    
    
    exit(234)
    root = "/home/ligongzhe/data/RAW/"
    data_list = os.listdir(root)
    
    # print(len(data_list))
    
    cnt = 0
    load_times = []

    for i in data_list:
        if cnt>1000:
            break
        cnt += 1
        st = time.time()
        data = np.load(os.path.join(root, i))
        # data = fastload(os.path.join(root, i))
        et = time.time()
        load_times.append(et-st)
        
        data = np.clip(data*(BIT24-1), 0, (BIT24-1)).astype(np.int32)
        cv2.imwrite("./test.tiff", data)
        
        exit(234)

    # 输出平均读取时间和每次的读取时间
    average_time = sum(load_times) / cnt
    print("npy load")
    print(f"平均读取.npy文件耗时: {average_time:.6f} 秒")
    print(f"{cnt}数量耗时: {sum(load_times):.6f} 秒")

    # print(f"每次读取时间列表: {load_times}")
