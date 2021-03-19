
import os, sys
import numpy as np


def azure_jsonval2npy(face_learned_img_npy_path, attri_list):
    gender, glasses, yaw, pitch, bald, beard, age, expression = -1
    face_attr_npy = np.zeros([len(attri_list), 8, 1])
    if not os.path.exists(face_learned_img_npy_path):
        os.makedirs(face_learned_img_npy_path)
    for i in range(len(attri_list)):
        gender, glasses, yaw, pitch, bald, beard, age, expression = attri_list[i]
        face_attr_npy[i] = np.expand_dims(np.array([gender,  glasses,  yaw,  pitch,  bald,  beard,  age,  expression]), axis = 1)
        try:
            np.save(face_learned_img_npy_path + '/attributes', face_attr_npy)
        except Exception as e:
            print(e)
        print(face_attr_npy)
        print(face_attr_npy.shape)




if __name__ == '__main__':
    # 男, 戴眼镜, 水平偏转, 上下偏转, 秃头, 胡子, 年龄, 微笑程度 = [1, 1, yaw, pitch, bald, beard, age, smile]
    attribuates_list = [[0, 0, 19.1, -5.9, 0.05, 0, 21, 0.983]]
    img_learned_path = '../newimg_data/'
    azure_jsonval2npy(img_learned_path, attribuates_list)
