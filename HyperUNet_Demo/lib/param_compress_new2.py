#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 22:25:30 2024

@author: dayang
"""

## save one dictionary, search for others
from tqdm import tqdm
import sys
import numpy as np
from scipy.spatial.distance import cityblock
from sklearn.neighbors import KDTree
import multiprocessing
import os
import matplotlib.pyplot as plt

def define_global(codebook_num, Uint_i, k, k_lossless):

    global side_length, K_lossless, dimension, Max, tree, Try, K
    n = codebook_num
    side_length = n + 1
    K_lossless = k_lossless
    K = k
    uint_i = Uint_i

    a = np.linspace(0, K - 1, K)
    b = 1 / (np.pi + a)

    Max_List_Path = f'./lib/uint{uint_i}_list.txt'
    Max = []
    with open(Max_List_Path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            if line[-1] == '\n':
                Max.append(int(line[:-1]))
            else:
                Max.append(int(line))

    dimension = [i for i in range(2, len(Max)+2)]

    R = 100   # 1/R : 步长

    Allval = np.arange(n)
    Allval = Allval.reshape((n,1))

    Cn = b/R  # 方向向量
    Cn = Cn.reshape((1,K))

    Target = np.random.uniform(0,1,K)
    Target = Target.reshape((1,K))
    i_target = 0

    Try = Allval @ Cn
    ## appy decimal function
    Try, integral_part = np.modf(Try) # 向量化

    # ## sort Try array so that close points have close index
    # from sklearn.neighbors import NearestNeighbors
    # neighbors_model = NearestNeighbors(n_neighbors=1000, algorithm='auto', metric='euclidean')
    # neighbors_model.fit(Try)
    # distances, indices = neighbors_model.kneighbors(Try)
    # Try = Try[indices[:, 1]]

    # from scipy.spatial import cKDTree
    # kdtree = cKDTree(Try)
    # distances, indices = kdtree.query(Try, k=len(Try), p=2)
    # Try = Try[indices[:, 1]]

    ## get a new tree for query
    tree = KDTree(Try, leaf_size=100)

    # Example arrays A and B
    # A = np.random.rand(1000, 5)
    Ba = 32
    Target = np.random.rand(Ba, K)



def Decode_Params(Save_CompressedResult_RootPath):

    print("#"*50,'\n',"Start Decoding",'\n',"#"*50)

    decode_params_list = []
    infor_path = Save_CompressedResult_RootPath + 'infor.bin'
    compressed_num_zero_path = Save_CompressedResult_RootPath + 'compressed_num_zero.bin'
    FloatInfor_path = Save_CompressedResult_RootPath + 'float.bin'
    Cha_path = Save_CompressedResult_RootPath + 'Cha.bin'
    Size_tar_path = Save_CompressedResult_RootPath + 'Size_tar.bin'

    infor = np.fromfile(infor_path, dtype=np.int32)
    compress_zeros = np.fromfile(compressed_num_zero_path, dtype=np.uint64)
    FloatInfor = np.fromfile(FloatInfor_path, dtype=np.float32).reshape([-1,2])
    Cha = np.fromfile(Cha_path, dtype=np.int32)
    Size_tar = np.fromfile(Size_tar_path, dtype=np.int32).reshape([-1,2])

    w = infor[0]
    N = infor[1]
    k_lossless = infor[2]

    l = Max[dimension.index(N)] + 1
    back_step_1 = decompression_2(compress_zeros, l, N)

    if w != 0:
        back_step_1 = back_step_1[:-w]

    r"""
        decode layers' params
    """

    # 获取 root文件夹中所有文件名中不带"_"的文件名
    file_names = os.listdir(Save_CompressedResult_RootPath)
    num_layers = len(file_names) - 5


    # 构建 layer_i 与 文件地址的dict
    dict_layer_files = {}
    for file_name in file_names:
        try:
            i_layer = int(file_name[:file_name.index('_')]) # 第几层的参数
            shape_str = file_name[file_name.index('_')+1 : -4].split('_')
            shape = []
            for j in shape_str:
                shape.append(int(j))
            shape = np.array(shape) # 参数的shape
            dict_layer_files[i_layer] = [Save_CompressedResult_RootPath + file_name ,shape]
        except:
            # except 说明是“非layer params”文件
            continue

    print(f"一共 {num_layers} layers，成功获取{len(list(dict_layer_files.values()))}个文件")

    #index_count = {}  # 模型所有index中的最大值
    if num_layers == len(list(dict_layer_files.values())):
        # 遍历到每一个文件名和shape
        for i in tqdm(range(num_layers)):
            # layer_param_index decode 为 float params
            cha = Cha[i]
            size_tar = tuple(Size_tar[i])
            mean = FloatInfor[i][0]
            max_abs = FloatInfor[i][1]

            if mean==0 and max_abs==0 and cha==0 and size_tar==tuple([0,0]): # 当时压缩失败的参数直接load
                layer_path = dict_layer_files[i][0]  # 文件地址
                layer_shape = dict_layer_files[i][1]  # 恢复的shape
                param1 = np.fromfile(layer_path, dtype=np.float32)
                param1 = np.array(param1).reshape(layer_shape)
                decode_params_list.append(param1)

            else:
                layer_path = dict_layer_files[i][0]  # 文件地址
                layer_shape = dict_layer_files[i][1] # 恢复的shape
                layer_param_int = np.fromfile(layer_path, dtype=np.uint64)
                layer_param_index = []
                for num in layer_param_int:
                    layer_param_index += back2node(num, Max[dimension.index(k_lossless)] + 1, k_lossless)


                # layer_param_index 为 整数index结果
                if int(back_step_1[i]) != 0:
                    layer_param_index = layer_param_index[:-int(back_step_1[i])]

                # param1 : 将整数index结果decode为新的float参数
                param1 = decompression_1d(np.array(layer_param_index).reshape([-1,1]), cha, size_tar, Try)
                param1 = restore_data(param1, mean, max_abs)
                param1 = np.array(param1).reshape(layer_shape)

                decode_params_list.append(param1)

    else:
        print("Something Wrong During Decoding!")

    print("#" * 50, '\n', "Finish Decoding", '\n', "#" * 50)
    return decode_params_list


def Save_Num_Zero(Num_Zero, root_dir, K_lossless):
    save_name_infor = "infor.bin"
    save_path_infor = root_dir + save_name_infor
    save_name_compressed_num_zero = "compressed_num_zero.bin"
    save_path_compressed_num_zero = root_dir + save_name_compressed_num_zero

    _, N = check_comp(Num_Zero) # 可压最大倍数

    side_length_zero = Max[dimension.index(N)] + 1
    new_Num_Zero, w = add_zero(np.array(Num_Zero), N)
    Reshape_new_Num_Zero = np.array(new_Num_Zero).reshape(-1, N).tolist()
    
    with multiprocessing.Pool(processes=8) as pool:
        #使用进程池的 starmap 方法并行处理列表中的每个元素和索引
        compress_zeros = pool.starmap(process, [[Reshape_new_Num_Zero[index], index, side_length_zero, len(Reshape_new_Num_Zero)] for index in range(len(Reshape_new_Num_Zero))])

    infor = np.array([w, N, K_lossless])
    if np.max(infor) > 2**8:
        print("There is an int in infor > int8!!!!!!")
    if np.max(infor) > 2**16:
        print("There is an int in infor > int16!!!!!!")
    
    np.array(compress_zeros).astype(np.uint64).tofile(save_path_compressed_num_zero)
    np.array(infor).astype(np.int32).tofile(save_path_infor)
    print("ok")




def checkdic1(Target,diction):
    ## Target [B,5]
    distances = np.linalg.norm(diction[:, np.newaxis, :] - Target, axis=2)
    closest_row_indices = np.argmin(distances, axis=0)
    return closest_row_indices[:, np.newaxis] ## (B,1)

def checkdic(Target,tree):
    ## Target [B,5]
    dist, closest_row_indices = tree.query(Target, k=1)
    return closest_row_indices ## (B,1)


from decimal import Decimal, getcontext

getcontext().prec = 30

def f(x,l):
    x = np.array(x)
    y = x % l
    return y


def find_center_coordinate(l, point):

    for i in range(len(point)):
        point[i] = min(int(point[i]), l - 1) + 0.5

    return point


def add_zero(list, K):
    n = len(list)
    if n % K != 0:
        num_zero = K - n % K
        new_list = list.tolist() + [0]*num_zero
    else:
        new_list = list
        num_zero = 0

    return new_list, num_zero


def compute_S(node,l):
    K = len(node)
    L = [1]
    for t in range(K-1):
        num = L[0]
        L.insert(0, num*l)

    getcontext().prec = 20

    L2 = np.array(L).reshape((1,len(L)))
    node2 = np.array(node).reshape((len(node),1))
    S2 = (L2 @ node2)[0][0]

    return S2

def back2node(S, l, K):
    l = int(l)
    L = [Decimal(l)]
    for t in range(K - 1):
        num = L[0]
        L.insert(0, num / l)
    b = L

    R = Decimal(0)
    for i in range(len(b)):
        R += b[i] ** 2
    R = R.sqrt()

    b_1 = b
    for i in range(len(b_1)):
        b_1[i] = b_1[i] / R

    step_size = R / l

    B = b_1
    for i in range(len(B)):
        B[i] = B[i] * step_size

    start = [0] * K
    for i in range(len(start)):
        start[i] = start[i] + (step_size / 2) * b_1[i]

    step_of_S = start  # x = f(S * B + start, l)
    for i in range(len(step_of_S)):
        step_of_S[i] = step_of_S[i] + S * B[i]
    x = f(step_of_S, l)

    node = find_center_coordinate(l, x)
    node = np.array(node) - np.array([0.5] * K)
    result = node.tolist()
    return result

def check_comp(inn):
    m = max(inn)  #当前集合中的最大数字
    for n in range(len(Max)):
        if Max[n] < m :
            max_dimension = dimension[n-1] #当前理论上的最大倍数
            break
    if not 'max_dimension' in locals():
        max_dimension = 65

    if max_dimension >= len(inn):
        return True, max_dimension
    else:
        return False, max_dimension


def decompression_2(Ready2Back, l, k_lossless):  # 将Result的list每个数字decode后输出为flatten list
    param1 = []

    for r in Ready2Back:
        p = back2node(r, l, k_lossless)
        param1.append(p)

    back_flattened_array = []
    for w in param1:
        back_flattened_array += w

    return back_flattened_array

def process(node, index, side_length_set, total_num):
    compute_S_result = compute_S(node, side_length_set)
    return compute_S_result

def process_dynamic(node, l):
    return compute_S(node, l+1)


def compress_decompre_scale_deep_2(inputx): ## input array for testing and recover the same parameter, numpy array type

    flattened_array = inputx.flatten()

    if len(flattened_array) % K_lossless != 0: # 补0
        new_array, num_zero = add_zero(flattened_array, K_lossless)
    else:
        new_array = flattened_array
        num_zero = 0

    new_array = np.array(new_array).astype(np.uint64)
    Ready2Compress = new_array.reshape(-1, K_lossless)
    Ready2Compress = Ready2Compress.tolist()


    with multiprocessing.Pool(processes=8) as pool:
        # 使用进程池的 starmap 方法并行处理列表中的每个元素和索引
        Result = pool.starmap(process, [[Ready2Compress[index], index, side_length, len(Ready2Compress)] for index in range(len(Ready2Compress))])

    return Result, num_zero


#%%
## Deal with matrix size of [500,100]

import time

def batch_compression(TargetX, Tree, Ba=128):
    bat,cha = TargetX.shape
    newch = int(np.ceil(cha/K))
    newba = int(np.ceil(bat/Ba))

    pad_size = newch*K - cha
    if pad_size > 0:
      TargetX = np.pad(TargetX, ((0, 0), (0, pad_size)), constant_values=0.5)

    output_idx = np.zeros((bat,newch))
    for i in range(newba):
        for j in range(newch):
            hi = i*Ba
            wi = j*K
            tmp_res = checkdic(TargetX[hi:hi+Ba,wi:wi+K],Tree)
            # print(tmp_res.shape)
            output_idx[hi:hi+Ba,j] = tmp_res.reshape(-1)

    return output_idx, (bat,cha)



def decompression(output_idx,size_tar,diction):
    bat,ch = output_idx.shape
    outputx = np.zeros((bat,ch*K))

    for c in range(ch):
        outputx[:,c*K:c*K+K] = diction[output_idx[:,c].astype(int)]

    return outputx[:,:size_tar[1]]

def batch_compression_1d(TargetX, tree, Ba=128): ## reshape to 2d, and then reshape back to 1d
    cha = len(TargetX)
    newch = int(np.ceil(cha/K))

    pad_size = newch*K - cha
    if pad_size > 0:
      TargetX = np.pad(TargetX, (0, pad_size), constant_values=0.5)

    TargetX = TargetX.reshape(-1, K)
    results, sizere =  batch_compression(TargetX, Tree=tree, Ba=128) ## results [n,1]

    return results, cha, sizere


def decompression_1d(output_idx,cha,sizere,Try):

    outputx = decompression(output_idx,sizere,diction=Try)
    # print(outputx.shape)
    outputx = outputx.flatten()[:cha]
    return outputx

def scale_data(data):
    # Step 1: Subtract the mean to make the average zero
    mean = np.mean(data)
    data_centered = data - mean
    # Step 2: Scale the data to be in the range [-0.5, 0.5]
    max_abs = np.max(np.abs(data_centered))
    scaled_data = data_centered / (2 * max_abs)  # Scale to [-0.5, 0.5]
    # Step 3: Shift the scaled data to have an average of 0.5
    scaled_data = scaled_data + 0.5

    return scaled_data, mean, max_abs

def restore_data(scaled_data, mean, max_abs):
    # Step 1: Shift the scaled data to remove the average of 0.5
    shifted_data = scaled_data - 0.5
    # Step 2: Restore the original scale
    restored_data = shifted_data * (2 * max_abs)
    # Step 3: Add back the mean
    original_data = restored_data + mean

    return original_data

def compress_decom_v2(inputx): ## most easy form to deal with all shapes, only need to deal with 1d
    ## scale the data to 0 - 1
# =============================================================================
#     max_value = inputx.max()
#     min_value = inputx.min()
#     #min_value, max_value = np.percentile(inputx, [.5, 99.5]) #[2.5, 97.5]
#     inputx = (inputx - min_value) / (max_value - min_value)
# =============================================================================
    inputx, mean, max_abs = scale_data(inputx)

    ori_shape = inputx.shape
    inputx = inputx.flatten()
    output_idx, cha, size_tar = batch_compression_1d(inputx, Tree=tree, Ba=128)
    param1 = decompression_1d(output_idx,cha,size_tar,diction=Try)
    param1 = param1.reshape(ori_shape)

    ## scale value back
# =============================================================================
#     param1 = param1 * (max_value - min_value) + min_value
# =============================================================================
    # Restore the scaled data
    param1 = restore_data(param1, mean, max_abs)
    return param1

from scipy import sparse
def compress_decom_v3(inputx): ## sparse matrix #most easy form to deal with all shapes, only need to deal with 1d
    ## sparse matrix compression
    
    ori_shape = inputx.shape
    inputx = inputx.flatten()
    inputx[np.where(inputx == 0)] = np.finfo(np.float32).tiny

    ## get sparse matrix
    sparse_in = sparse.csr_matrix(inputx)
    sp_data = sparse_in.data   ## data for compression
    sp_idx = sparse_in.indices ## idx for compression


    sp_data, mean, max_abs = scale_data(sp_data)

    # output_idx ：小数压缩后的index结果
    output_idx, cha, size_tar = batch_compression_1d(sp_data, Tree=tree, Ba=128)

    # Result : 整数压缩后的结果
    Result, num_zero = compress_decompre_scale_deep_2(output_idx.T)

    # param1 : 将整数index结果decode为新的float参数
    param1 = decompression_1d(output_idx,cha,size_tar,diction=Try)
    param1 = restore_data(param1, mean, max_abs)
    
    sparse_in.data = np.array(param1)
    param1 = sparse_in.A ## get original matrix
     
    param1 = param1.reshape(ori_shape)
   
    return param1, Result, num_zero, cha, size_tar, mean, max_abs, output_idx
    
    

def batch_compression_3d(TargetX, tree, Ba=128): ## reshape to 2d, and then reshape back to 1d
    # Reshape the 3D array to 2D
    size3d = TargetX.shape
    TargetX = TargetX.reshape(-1, TargetX.shape[-1])

    ## compression
    output_idx, size2d = batch_compression(TargetX, Tree=tree, Ba=128)

    return output_idx, size3d, size2d


def decompression_3d(output_idx,size3d,size2d,Try):
    outputx = decompression(output_idx,size2d,diction=Try)
    outputx = outputx.reshape(size3d)
    return outputx


def compress_decompre(inputx): ## input array for testing and recover the same parameter, numpy array type
    """
    input: inputx, 1d,2d,3d array
    output:, param1, array with same shape
    """
    ## scale the data to 0 - 1
    max_value = inputx.max()
    min_value = inputx.min()
    inputx = (inputx - min_value) / (max_value - min_value)

    #print(inputx.ndim)
    if inputx.ndim == 1:
        output_idx, cha, size_tar = batch_compression_1d(inputx, Tree=tree, Ba=128)
        #print(output_idx.shape)
        param1 = decompression_1d(output_idx,cha,size_tar,diction=Try)
    elif inputx.ndim == 2:
        output_idx,size_tar = batch_compression(inputx, Tree=tree, Ba=128)
        #print(output_idx.shape)
        param1 = decompression(output_idx,size_tar,diction=Try)
    elif inputx.ndim == 3:
        output_idx, size3d, size2d = batch_compression_3d(inputx, Tree=tree, Ba=128)
        #print(output_idx.shape)
        param1 = decompression_3d(output_idx,size3d,size2d,diction=Try)
    elif inputx.ndim == 4:
        b,c,h,w = inputx.shape
        inputx3 = inputx.reshape((b,c,-1))
        ## three dims
        output_idx, size3d, size2d = batch_compression_3d(inputx3, Tree=tree, Ba=128)
        #output_idx = compress_decompre_scale_deep_2(output_idx.T) ## level 2 compression
        #output_idx = output_idx.T
        param11 = decompression_3d(output_idx,size3d,size2d,diction=Try)
        param1 = param11.reshape((b,c,h,w))
    else:
        param1 = 0
        print('error dimension!')
        
    ## scale value back
    param1 = param1 * (max_value - min_value) + min_value
    return param1

def compress_decompre_scale_deep(inputx): ## input array for testing and recover the same parameter, numpy array type
    """
    input: inputx, 1d,2d,3d array
    output:, param1, array with same shape
    """
    ## scale the data to 0 - 1, change the idx value mapping since it's very big
    max_value = inputx.max() # +10000
    min_value = inputx.min() # -10000
    inputx = (inputx - min_value) / (max_value - min_value)

    print(inputx.ndim)
    if inputx.ndim == 1:
        output_idx, cha, size_tar = batch_compression_1d(inputx, Tree=tree_l2, Ba=128)
        print(output_idx.shape)
        param1 = decompression_1d(output_idx,cha,size_tar,diction=Try_l2)
    if inputx.ndim == 2:
        output_idx,size_tar = batch_compression(inputx, Tree=tree_l2, Ba=128)
        print(output_idx.shape)
        param1 = decompression(output_idx,size_tar,diction=Try_l2)
    if inputx.ndim == 3:
        output_idx, size3d, size2d = batch_compression_3d(inputx, Tree=tree_l2, Ba=128)
        print(output_idx.shape)
        param1 = decompression_3d(output_idx,size3d,size2d,diction=Try_l2)

    ## scale value back
    param1 = param1 * (max_value - min_value) + min_value
    return param1

def compress_decompre_l2(inputx): ## input array for testing and recover the same parameter, numpy array type
    """
    input: inputx, 1d,2d,3d array
    output:, param1, array with same shape
    """
    ## scale the data to 0 - 1
    max_value = inputx.max()
    min_value = inputx.min()
    inputx = (inputx - min_value) / (max_value - min_value)

    ## scale to 0.2-0.8 to avoid jump
    gap = 0.1
    inputx = gap + inputx * (1 - 2*gap) ## scale from 0-1

    if inputx.ndim == 1:
        output_idx, cha, size_tar = batch_compression_1d(inputx, Tree=tree, Ba=128)
        output_idx = compress_decompre_scale_deep_2(output_idx.T) ## level 2 compression, Transpose, to compression on another dimension
        output_idx = output_idx.T
        param1 = decompression_1d(output_idx,cha,size_tar,diction=Try)
    elif inputx.ndim == 2:
        output_idx,size_tar = batch_compression(inputx, Tree=tree, Ba=128)
        print(output_idx)
        output_idx = compress_decompre_scale_deep_2(output_idx.T) ## level 2 compression
        output_idx = output_idx.T
        print(output_idx)
        param1 = decompression(output_idx,size_tar,diction=Try)
    elif inputx.ndim == 3:
        output_idx, size3d, size2d = batch_compression_3d(inputx, Tree=tree, Ba=128)
        output_idx = compress_decompre_scale_deep_2(output_idx.T) ## level 2 compression
        output_idx = output_idx.T
        param1 = decompression_3d(output_idx,size3d,size2d,diction=Try)
    elif inputx.ndim == 4:
        b,c,h,w = inputx.shape
        inputx3 = inputx.reshape((b,c,-1))
        ## three dims
        output_idx, size3d, size2d = batch_compression_3d(inputx3, Tree=tree, Ba=128)
        output_idx = compress_decompre_scale_deep_2(output_idx.T) ## level 2 compression
        output_idx = output_idx.T
        param11 = decompression_3d(output_idx,size3d,size2d,diction=Try)
        param1 = param11.reshape((b,c,h,w))
    else:
        param1 = 0
        print('error dimension!')
    param1 = (param1 - gap) / (1 - 2*gap) ## from 0.2-0.8 to 0-1
    param1 = param1 * (max_value - min_value) + min_value
    return param1


if __name__=="__main__":
    check_comp([0,1])