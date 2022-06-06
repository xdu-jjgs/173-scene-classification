import h5py
import numpy as np
import scipy.io as sio

'''
training.h5:	training data containing SEN1, SEN2 patches and label
N = 352366
sen1:	N*32*32*8
sen2:	N*32*32*10
label:	N*17 (one-hot coding)

validation.h5:  validation data containing similar SEN1, SEN2, and label
M = 24119
sen1:  	M*32*32*8 
sen2:  	M*32*32*10
label: 	M*17 (one-hot coding)

L = 24188
sen1:  	L*32*32*8
sen2:  	L*32*32*10
label:  L*17 (one-hot coding)
'''


def SelectSamples(path, file_name, sample_num, class_interest=None):
    # read data
    file = h5py.File(path + file_name, 'r')

    # change the one-hot label into class num label
    print('loading the label')
    label = np.array(file['label'])  # shape: [352366,17]
    label_class = np.argmax(label, axis=1)
    total_sample = len(class_interest) * sample_num
    label_select = np.zeros([total_sample, 17])
    for class_index in class_interest:
        index_list = np.where(label_class == class_index)
        print(index_list[0].shape)
        # index_list = tuple(list(index_list[0])[:sample_num])
        index_list = index_list[0][:sample_num]
        print(np.array(index_list).shape)
        iter = int(np.where(class_interest == class_index)[0])
        print('iter:', iter, 'class:', class_index)
        label_select[iter*sample_num:(iter+1)*sample_num, :] = label[index_list, :]
    print('finish label selecting')

    # 内存占用太大了没法同时读s1 s2做处理 所以分开写两个循环
    print('loading the sar data')
    s1 = np.array(file['sen1'])  # shape: [352366,32,32,8]
    s1_select = np.zeros([total_sample, 32, 32, 8])
    for class_index in class_interest:
        index_list = np.where(label_class == class_index)
        # print(index_list[0].shape)
        index_list = index_list[0][:sample_num]
        # print(np.array(index_list).shape)
        iter = int(np.where(class_interest == class_index)[0])
        print('iter:', iter, 'class:', class_index)
        s1_select[iter*sample_num:(iter+1)*sample_num, :, :, :] = s1[index_list, :, :, :]
    print('finish sar data selecting')
    s1 = []  # 释放内存

    print('loading the msi data')
    s2 = np.array(file['sen2'])  # shape: [352366,32,32,10]
    s2_select = np.zeros([total_sample, 32, 32, 10])
    for class_index in class_interest:
        index_list = np.where(label_class == class_index)
        index_list = index_list[0][:sample_num]
        # iter = int(np.where(class_interest == class_index))
        iter = int(np.where(class_interest == class_index)[0])
        print('iter:', iter, 'class:', class_index)
        s2_select[iter*sample_num:(iter+1)*sample_num, :, :, :] = s2[index_list, :, :, :]
    print('finish msi data selecting')
    s2 = []  # 释放内存

    # print('loading the sar data')
    # s1 = np.array(file['sen1'])  # shape: [352366,32,32,8]
    # shuffle_ix = np.random.permutation(np.arange(s1.shape[0]))  # total_num = s1.shape[0]随机打乱
    # shuffle_ix = shuffle_ix[:sample_num]  # 取前sample_num个索引
    # s1_selected = s1[shuffle_ix]
    # print('finish sar data selecting')
    # s1 = []  # 释放内存
    #
    # print('loading the msi data')
    # s2 = np.array(file['sen2'])  # shape: [352366,32,32,10]
    # s2_selected = s2[shuffle_ix]
    # print('finish msi data selecting')
    # s2 = []  # 释放内存
    #
    # label = np.array(file['label'])  # shape: [352366,17]
    # label_selected = label[shuffle_ix]  # 此处的label是17维one-hot形式
    # label = []  # 释放内存
    return s1_select, s2_select, label_select


def LabelTransform(label, class_intrest=None):
    # label shape: [sample_num, 17]
    label_class = np.argmax(label, axis=1)  # 如果在选择样本时做过了，此处可以注释掉
    if class_intrest is None:
        return label_class, 17
    else:  # 如果对类别进行了挑选，则重新编号01234
        for i in range(len(class_intrest)):
            c = class_intrest[i]
            # TODO：这个地方注意下，逻辑上应该是i+1 因为main里面的onehot操作会-1，但是好像跑起来实际不影响
            label_class[np.where(label_class == c)] = i
        return label_class, len(class_intrest)


if __name__ == '__main__':
    data_dir = "E:/dataset/SAR-MSI_dataset/So2Sat_LCZ42/"
    # data_name = 'training.h5'
    # save_path = 'data/data_train.mat'
    # sample_num = 2048  # 2^11 这里是每类的样本数
    data_name = 'testing.h5'
    save_path = 'data/data_test.mat'
    sample_num = 128  # 2^7
    class_interest = np.array([0, 9, 10, 13, 14, 16])
    s1, s2, label = SelectSamples(data_dir, data_name, sample_num, class_interest)
    label, class_num = LabelTransform(label, class_interest)
    print('finish label transforming')
    print(s1.shape, s2.shape, label.shape)
    sio.savemat(save_path, {'sar': s1, 'msi': s2, 'label': label, 'class_num': class_num})
    print(label[:10])  # [13 11 10  1  1 10  7 16 15 10]
