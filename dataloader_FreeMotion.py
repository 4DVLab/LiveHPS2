import os
import numpy as np
import transforms3d
from scipy.io import loadmat
from pytorch3d.transforms import *
import math
from tqdm import tqdm
import torch

def rotateBody(RzBody, pelvisRotVec):
    angle = np.linalg.norm(pelvisRotVec)
    Rpelvis = transforms3d.axangles.axangle2mat(pelvisRotVec / angle, angle)
    globRotMat = np.dot(RzBody, Rpelvis)
    R90 = transforms3d.euler.euler2mat(np.pi / 2, 0, 0)
    globRotAx, globRotAngle = transforms3d.axangles.mat2axangle(np.dot(R90, globRotMat))
    globRotVec = globRotAx * globRotAngle
    return globRotVec

root_path = 'PATH_TO_SURREAL/surreal/cmu/train/run1/'
path2 ='PATH_TO_FREEMOTION/surreal/pc_2/train/run1/'

path1 = path2.replace('pc_2','pc_1')
path3 = path2.replace('pc_2','pc_3')
seqs_name = os.listdir(path2)
seqs_name.sort()
rot = torch.tensor([[-1,0,0],[0,0,-1],[0,-1,0]]).float()
rot1 = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
rot2 = np.array([[0,1,0],[-1,0,0],[0,0,1]])
datas = []
for seq_name in tqdm(seqs_name):
    data1 = np.load(path1+seq_name,allow_pickle=True)
    data2 = np.load(path2+seq_name,allow_pickle=True)
    data3 = np.load(path3+seq_name,allow_pickle=True)
    gt_path_old = ''
    index=0
    for i in range(len(data1)):
        gt_path_new = root_path+seq_name.split('.')[0]+'/'+data1[i]['target_name'].split('/')[-2]
        if gt_path_new!=gt_path_old:
            gt_path_old = gt_path_new
            gt_info = loadmat(gt_path_new+'.mat')
            pose = gt_info['pose'].T #T,72
            shape = gt_info['shape'].T #T,10
            zrot = np.array(gt_info['zrot'])
            index=0
        zs = zrot[index*3:index*3+3]
        ps = pose[index*3:index*3+3]
        ss = shape[index*3:index*3+3]
        for j in range(3):
            p = ps[j]
            z = zs[j]
            RzBody = np.array(((math.cos(z), -math.sin(z), 0),
                    (math.sin(z), math.cos(z), 0),
                    (0, 0, 1)))
            p[:3] = rotateBody(RzBody,p[:3])
            p[:3] = matrix_to_axis_angle(torch.matmul(rot,axis_angle_to_matrix(torch.from_numpy(p[:3])))).numpy().reshape(3)
        
        info = {
            'pc_1':data1[i]['points'],
            'pc_2':np.matmul(rot1.T,data2[i]['points'].T).T,
            'pc_3':np.matmul(rot2.T,data3[i]['points'].T).T,
            'gt': ps,
            'shape': ss,
            'T_1':data1[i]['trans'],
            'T_2':np.matmul(rot1.T,data2[i]['trans']),
            'T_3':np.matmul(rot2.T,data3[i]['trans']),
            'rot1':rot1,
            'rot2':rot2,
            'seq_path':gt_path_new
        }
        index+=1
        datas.append(info)
print(len(datas))
import pickle
with open('./FreeMotion_train.pkl','wb') as f:
    pickle.dump(datas,f)