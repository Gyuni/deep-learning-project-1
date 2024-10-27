import os.path
import gzip
import pickle
import os
import numpy as np
from PIL import Image
import pandas as pd

key_file = {
    'train_img_dir':'/train_img',
    'train_label':'train_label.csv',
    'test_img_dir':'/test_img',
    'test_label':'test_label.csv'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/glaucoma.pkl"

train_num = 800
test_num = 220
image_dimension = 3
image_length = 64
img_dim = (image_dimension, image_length, image_length)
img_size = image_length*image_length

def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")
    labels_df = pd.read_csv(file_path)
    labels = labels_df['binaryLabels'].values
    print("Done, count: ", len(labels))

    return labels

def _load_img(dir):
    file_path = dataset_dir + dir
    
    print("Converting " + dir + " to NumPy Array ...")
    images = []
    count = 0
    for filename in os.listdir(file_path):
        if filename.endswith('.jpg'):
            count += 1
            img_path = os.path.join(file_path, filename)
            img = Image.open(img_path).resize((image_length, image_length))
            img = np.array(img)
            images.append(img)

    data = np.array(images)
    data = data.reshape(-1, img_size)
    print("Done, count: ", count)
    
    return data
    
def _convert_numpy():
    dataset = {}
    dataset['train_img'] =  _load_img(key_file['train_img_dir'])
    dataset['train_label'] = _load_label(key_file['train_label'])    
    dataset['test_img'] = _load_img(key_file['test_img_dir'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    
    return dataset

def init_glaucoma():
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T
    

def load_glaucoma(normalize=True, flatten=True, one_hot_label=False):
    """glaucoma 데이터셋 읽기
    
    Parameters
    ----------
    normalize : 이미지의 픽셀 값을 0.0~1.0 사이의 값으로 정규화할지 정한다.
    one_hot_label : 
        one_hot_label이 True면、레이블을 원-핫(one-hot) 배열로 돌려준다.
        one-hot 배열은 예를 들어 [0,0,1,0,0,0,0,0,0,0]처럼 한 원소만 1인 배열이다.
    flatten : 입력 이미지를 1차원 배열로 만들지를 정한다. 
    
    Returns
    -------
    (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
    """
    # if not os.path.exists(save_file):
    #     init_glaucoma()

    init_glaucoma()
        
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
            
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])    
    
    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, image_dimension, image_length, image_length)

    # type: (<class 'numpy.ndarray'>, <class 'numpy.ndarray'>, <class 'numpy.ndarray'>, <class 'numpy.ndarray'>)
    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 


if __name__ == '__main__':
    init_glaucoma()
