import os,glob, tarfile
import torch
from torch.autograd import Variable
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
Note: The meaning of batch_size in PTB is different from that in MNIST example. In MNIST, 
batch_size is the # of sample data that is considered in each iteration; in PTB, however,
it is the number of segments to speed up computation. 

The goal of PTB is to train a language model to predict the next word.
"""


def make_targz(output_filename, source_dir):
    """
    一次性打包目录为tar.gz
    :param output_filename: 压缩文件名
    :param source_dir: 需要打包的目录
    :return: bool
    """
    try:
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))

        return True
    except Exception as e:
        print(e)
        return False


#T_train: history data length
#T_predict: predict data length
#手动分割数据，数据处理到train长度，对应预测长度
#线性归一化
def generate_Seq_npy_data(filePath, savePath,  T_train, T_predict):

    T_Window = T_train + T_predict #预测多长时间，默认一周7*24
    df = pd.read_csv(filePath)
    id = list(df['id'])
    readtime = list(df['readtime'])
    reading = list(df['reading'])


    np_alldata =  np.array(reading)
    all_data_max = np_alldata.max()
    all_data_min = np_alldata.min()

    norm_all_data = (np_alldata - all_data_min)/(all_data_max - all_data_min)

    trainPath = savePath + 'AllData/'

    if  not os.path.exists(trainPath):
        os.makedirs(trainPath)

    assert (len(id) == len(readtime) == len(norm_all_data))
    dataset_size = len(id)
    id_data_list = [] #id集合 每个id一行存所有点 
    tp_data = []   #临时存放当前id的所有数据，一个用户用完就清空
    id_flag = 0    # 标志位，循环中标定当前位置
    # id data 组合成list
    for i in range(len(id)):#遍历所有数据，788400个
        if id[i] == id_flag:#如果id对应上csv文件id, 存入tp_data中
            tp_data.append(norm_all_data[i])
            # print()
        else:#用户id数据寻找完，存当前用户数据到id_data_list，并设置下一个用户标志
            id_flag+=1#标志位+1，表示搜索下一个用户
            id_data_list.append(tp_data)#当前用户数据存入到id_data_list
            tp_data = []#清空上一个用户数据
            # print('上一个用户，id ' + str(id[i]-1)+'  has done.....，最后一个数据为：')
            # print(id_data_list[id_flag-1][-1])

            tp_data.append(norm_all_data[i])#当前用户第一个数据存入
            # print('当前用户id '+str(id_flag)+ '第一个数据是 ')
            # print(reading[i])
            # print()
    id_data_list.append(tp_data)#当前用户数据存入到id_data_list
    print('id assigning is done...')

    tp_data = []
    wind_data_list = []
    for i in range(len(id_data_list)):
        for j in range(len(id_data_list[i])):
            if j + T_Window >= len(id_data_list[i]):
                print('Processing done......T_start + T_length > T_Window')
                break
            else:
               tp_data = id_data_list[i][j:T_Window+j]
            #    print(id_data_list[i][T_Window+j])
               wind_data_list.append(tp_data)
               tp_data = []
        print('id windows split' +str(i) + ' has done.')

    data_length = len(wind_data_list)
    # lamnda_train_split = int(data_length * 0.8)
    # train_data = wind_data_list[0 : lamnda_train_split]
    # test_data = wind_data_list[lamnda_train_split : data_length]
    # for i in range(len(train_data)):
    #     tp_npy_name = i
    #     tp_savePath = trainPath + f'{tp_npy_name:05}' + '.npy'
    #     np.save(tp_savePath , train_data[i])

    # for i in range(len(test_data)):
    #     tp_npy_name = i
    #     tp_savePath = testPath + f'{tp_npy_name:05}' + '.npy'
    #     np.save(tp_savePath , test_data[i])
    
    for i in range(len(wind_data_list)):
        tp_npy_name = i
        tp_savePath = trainPath + f'{tp_npy_name:05}' + '.npy'
        np.save(tp_savePath , wind_data_list[i])
    return None

#零均值归一化/Z-score标准化
def generate_Seq_npy_data_Z(filePath, savePath,  T_train, T_predict):

    T_Window = T_train + T_predict #预测多长时间，默认一周7*24
    df = pd.read_csv(filePath)
    id = list(df['id'])
    readtime = list(df['readtime'])
    reading = list(df['reading'])


    np_alldata =  np.array(reading)
    all_data_mean = np_alldata.mean()
    all_data_std = np_alldata.std()

    norm_all_data = (np_alldata - all_data_mean)/all_data_std

    trainPath = savePath + 'AllData/'

    if  not os.path.exists(trainPath):
        os.makedirs(trainPath)

    assert (len(id) == len(readtime) == len(norm_all_data))
    dataset_size = len(id)
    id_data_list = [] #id集合 每个id一行存所有点 
    tp_data = []   #临时存放当前id的所有数据，一个用户用完就清空
    id_flag = 0    # 标志位，循环中标定当前位置
    # id data 组合成list
    for i in range(len(id)):#遍历所有数据，788400个
        if id[i] == id_flag:#如果id对应上csv文件id, 存入tp_data中
            tp_data.append(norm_all_data[i])
            # print()
        else:#用户id数据寻找完，存当前用户数据到id_data_list，并设置下一个用户标志
            id_flag+=1#标志位+1，表示搜索下一个用户
            id_data_list.append(tp_data)#当前用户数据存入到id_data_list
            tp_data = []#清空上一个用户数据
            # print('上一个用户，id ' + str(id[i]-1)+'  has done.....，最后一个数据为：')
            # print(id_data_list[id_flag-1][-1])

            tp_data.append(norm_all_data[i])#当前用户第一个数据存入
            # print('当前用户id '+str(id_flag)+ '第一个数据是 ')
            # print(reading[i])
            # print()
    id_data_list.append(tp_data)#当前用户数据存入到id_data_list
    print('id assigning is done...')

    tp_data = []
    wind_data_list = []
    for i in range(len(id_data_list)):
        for j in range(len(id_data_list[i])):
            if j + T_Window >= len(id_data_list[i]):
                print('Processing done......T_start + T_length > T_Window')
                break
            else:
               tp_data = id_data_list[i][j:T_Window+j]
            #    print(id_data_list[i][T_Window+j])
               wind_data_list.append(tp_data)
               tp_data = []
        print('id windows split' +str(i) + ' has done.')

    data_length = len(wind_data_list)
    # lamnda_train_split = int(data_length * 0.8)
    # train_data = wind_data_list[0 : lamnda_train_split]
    # test_data = wind_data_list[lamnda_train_split : data_length]
    # for i in range(len(train_data)):
    #     tp_npy_name = i
    #     tp_savePath = trainPath + f'{tp_npy_name:05}' + '.npy'
    #     np.save(tp_savePath , train_data[i])

    # for i in range(len(test_data)):
    #     tp_npy_name = i
    #     tp_savePath = testPath + f'{tp_npy_name:05}' + '.npy'
    #     np.save(tp_savePath , test_data[i])
    
    for i in range(len(wind_data_list)):
        tp_npy_name = i
        tp_savePath = trainPath + f'{tp_npy_name:05}' + '.npy'
        np.save(tp_savePath , wind_data_list[i])
    return None

#T_train: history data length
#T_predict: predict data length
#系统分割数据，数据处理到id 对一个序列
def generate_Seq_npy_data_auto(filePath, savePath,  T_train, T_predict):

    T_Window = T_train + T_predict #预测多长时间，默认一周7*24
    df = pd.read_csv(filePath)
    id = list(df['id'])
    readtime = list(df['readtime'])
    reading = list(df['reading'])

    trainPath = savePath + 'id_train_auto/'

    if  not os.path.exists(trainPath):
        os.makedirs(trainPath)

    assert (len(id) == len(readtime) == len(reading))
    dataset_size = len(id)
    id_data_list = [] #id集合 每个id一行存所有点 
    tp_data = []   #临时存放当前id的所有数据，一个用户用完就清空
    id_flag = 0    # 标志位，循环中标定当前位置
    # id data 组合成list
    for i in range(len(id)):#遍历所有数据，788400个
        if id[i] == id_flag:#如果id对应上csv文件id, 存入tp_data中
            tp_data.append(reading[i])
            # print()
        else:#用户id数据寻找完，存当前用户数据到id_data_list，并设置下一个用户标志
            id_flag+=1#标志位+1，表示搜索下一个用户
            id_data_list.append(tp_data)#当前用户数据存入到id_data_list
            tp_data = []#清空上一个用户数据
            # print('上一个用户，id ' + str(id[i]-1)+'  has done.....，最后一个数据为：')
            # print(id_data_list[id_flag-1][-1])

            tp_data.append(reading[i])#当前用户第一个数据存入
            # print('当前用户id '+str(id_flag)+ '第一个数据是 ')
            # print(reading[i])
            # print()
    id_data_list.append(tp_data)#当前用户数据存入到id_data_list
    print('id assigning is done...')

      
    for i in range(len(id_data_list)):
        tp_npy_name = i
        tp_savePath = trainPath + f'{tp_npy_name:05}' + '.npy'
        np.save(tp_savePath , id_data_list[i])
    return None


def generate_Seq_csv_data(path):
    train_list = glob.glob(path + 'AllData/*')
    # test_list = glob.glob(path + 'test/*')
    name_to_list = {'Sequence': train_list}
    df = pd.DataFrame(data=name_to_list)
    df.to_csv(path+'/all_data.csv')

    # name_to_list = {'Sequence': test_list}
    # df = pd.DataFrame(data=name_to_list)
    # df.to_csv(path+'/test.csv')
    
    print('train csv is done...')
    return None

def get_data_from_csv(syn_dir=None, read_from_csv=None, validation_split=0, test_split=0):
    df = pd.read_csv(read_from_csv)
    T_list = list(df['Sequence'])
    dataset_size = len(T_list)
    
    test_count = int(test_split * dataset_size)
    train_count = dataset_size - test_count

    # Build custom datasets
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    full_dataset = T_Dataset(T_list, transform)
    # TODO: This will vary dataset run-to-run
    # Shall we just split manually to ensure run-to-run train-val dataset is same?
    train_dataset, test_dataset = random_split(full_dataset, [train_count, test_count])
    return train_dataset, test_dataset

class T_Dataset(Dataset):
    def __init__(self, T_list, transform=None):

        self.T_list = T_list

        self.dataset_len = len(self.T_list)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])


    def __getitem__(self, index):
        loadData = np.load(self.T_list[index])
        loadData = loadData.astype(np.float32)


        return loadData, self.T_list[index]

    def __len__(self):
        return self.dataset_len


def point_plot(targets, output, path, epoch, bix):
    x = np.linspace(0, 1, 168)
    gt = targets[0].detach().cpu()
    pred = output[0].detach().cpu()
    plt.plot(x, pred, c='#526922', ls='-.')
    plt.plot(x, gt, c='#ff3508', ls='-.')

    plt.savefig(path + str(epoch) + '_' + str(bix)+ '_Comp.png')#保存图片

    # plt.show()
    plt.close()
    
    return None