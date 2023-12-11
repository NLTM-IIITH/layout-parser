"""
HEADERS
"""

import torch
import numpy as np
import os
# Let's pick the desired backend
# os.environ['USE_TF'] = '1'
os.environ['USE_TORCH'] = '1'
import matplotlib.pyplot as plt
from doctr.models import ocr_predictor
from collections import OrderedDict
from doctr.io import DocumentFile
from .helper import doctr_predictions,PREDICTOR_V2,convert_geometry_to_bbox

import json
import os
import shutil
import cv2
import h5py
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from tempfile import TemporaryDirectory
from server.modules.core.config import TEXT_ATTB_MODEL_PATH as BASE_MODEL_PATH

from os.path import basename, join
from .models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""
End
"""



"""
Model class for multiclass classification
"""
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(7*50*128, 128)  
        self.fc2 = nn.Linear(128, 4)     
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 7*50*128)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
"""
End
"""

"""
Operations class on the input image to resize it to 56*400 
"""

# processing images without changing aspect ratio
class ProcessImageBBox:
    def aspect_conservative_resize(self,orignal_image,height=170,width=1200):
        w = int(width)
        h = int(orignal_image.shape[0]*(width/orignal_image.shape[1]))
        if h>height:
            w=int(height*orignal_image.shape[1]/orignal_image.shape[0])
            h = height
        return cv2.resize(orignal_image,(w,h),interpolation=cv2.INTER_LINEAR)

    def aspect_conservative_resize_height(self,orignal_image,height=170,width=1200):
        h = int(height)
        w = int(orignal_image.shape[1]*(height/orignal_image.shape[0]))
        if w>width:
            h=int(width*orignal_image.shape[0]/orignal_image.shape[1])
            w = width
        return cv2.resize(orignal_image,(w,h),interpolation=cv2.INTER_LINEAR)

    def centralizer(self,orignal_image,height=170,width=1200):
        pre_processed_image = orignal_image
        if orignal_image.shape[1]>width:
            pre_processed_image = self.aspect_conservative_resize(orignal_image,height,width)
        elif orignal_image.shape[0]>height:
            pre_processed_image = self.aspect_conservative_resize_height(orignal_image,height,width)
        plain_image = np.zeros((height,width),dtype=np.float32)
        plain_image.fill(255)
        width_centering_factor = (plain_image.shape[1] - pre_processed_image.shape[1])//2
        height_centering_factor = (plain_image.shape[0] - pre_processed_image.shape[0])//2  
        plain_image[height_centering_factor:pre_processed_image.shape[0]+height_centering_factor,width_centering_factor:pre_processed_image.shape[1]+width_centering_factor] = pre_processed_image[:,:]

        return plain_image

"""
End
"""


"""
JSONhelper class generate and process JSON files.
"""
class JSONhelper:
    def __init__(self,temp_file_path):
        self.filename=f'{temp_file_path}/bbox_pixels.hdf5'
        self.json_file_1_path = f'{temp_file_path}/bbox_info.json'
        self.json_file_2_path = f'{temp_file_path}/rev_map.json'

    def retrieve_json(self):
        with open(self.json_file_1_path,'r') as file:
            bbox_info = json.load(file)
        with open(self.json_file_2_path,'r') as file:
            rev_map = json.load(file)
        return bbox_info,rev_map
    
    def model_input_retrieve(self):
        with h5py.File(self.filename,'r') as hf:
            name = self.filename.split('/')[-1].split('.')[-2]
            inputs = hf[name][:]
            inputs = np.array(inputs)
        return inputs
    
    def generate_bbox_json_files(self,image_location,temp_file_path):
        json_dict = {}
        reverse_map={}
        bbox_pixels=[]

        temp2 = TemporaryDirectory(prefix="img")
        folder_path = image_location

        if os.path.isfile(image_location):
            shutil.copy(image_location,temp2.name)
            folder_path = temp2.name
        
        count=0
        processor = ProcessImageBBox()

        for img in os.listdir(folder_path):
            image = cv2.imread(os.path.join(folder_path,img),cv2.IMREAD_GRAYSCALE)
            bbox_prediction = doctr_predictions(os.path.join(folder_path,img))
            json_dict[img] = []

            for w in bbox_prediction[0]:
                word = np.array(image[w[1]:w[3],w[0]:w[2]])
                if(word.shape[0]>0 and word.shape[1]>0):
                    json_dict[img].append({})
                    json_dict[img][len(json_dict[img])-1]['bb_dim'] = [w[0],w[1],w[2],w[3]]
                    json_dict[img][len(json_dict[img])-1]['bb_ids'] = []
                    json_dict[img][len(json_dict[img])-1]['bb_ids'].append({})
                    json_dict[img][len(json_dict[img])-1]['bb_ids'][len(json_dict[img][len(json_dict[img])-1]['bb_ids'])-1]['id'] = count

                    reverse_map[str(count)]={}
                    reverse_map[str(count)]['file'] = img
                    reverse_map[str(count)]['index'] = len(json_dict[img])-1

                    bbox_pixels.append(processor.centralizer(word,height=56,width=400))
                    count+=1

        with open(f'{temp_file_path}/bbox_info.json','w') as file:
            json.dump(json_dict,file)

        with open(f'{temp_file_path}/rev_map.json','w') as file:
            json.dump(reverse_map,file)

        with h5py.File(f'{temp_file_path}/bbox_pixels.hdf5', 'w') as f:
            f.create_dataset('bbox_pixels',data=bbox_pixels)



"""
Minibatch Generator for inference :
"""

class InferenceImageDataset(Dataset):
    def __init__(self,image_list,batch_size=64,transform=None):
        """
        Args:
            image_list (list): List of image paths.
            transform (callable, optional): Optional transform to be applied on an image.
            batch_size = Batch size to be fed one at a time in model.
        """
        self.image_list = image_list
        self.transform = transform
        self.idx = 0
        self.batch_size=batch_size

    def __len__(self):
        return (len(self.image_list) + self.batch_size-1)//(self.batch_size)

    def __getitem__(self, idx):
        batch_images = []
        for i in range(self.idx, min(self.idx + self.batch_size,self.idx+len(self.image_list)-(self.idx))):
            image = self.image_list[i]
            image = image.astype('uint8')

            if self.transform:
                image = self.transform(image)
            batch_images.append(image)
        self.idx += self.batch_size
        
        if len(batch_images)>0:
            return torch.stack(batch_images, dim=0)
        if len(batch_images)==0:
            return None
        
"""
End
"""



"""
Model Train/Test class provides the output for visualization
"""

class Model:
    def __init__(self):
        self.base_model_path= BASE_MODEL_PATH
        self.model_instance = CustomCNN()

    def get_model(self):
        pretrained_model = torch.load(self.base_model_path)
        state_pretrained= pretrained_model.module.state_dict()
        self.model_instance.load_state_dict(state_pretrained)
        self.model_instance = self.model_instance.to(device)
    
    def predict(self,temp_file_path):
        json_helper = JSONhelper(temp_file_path=temp_file_path)
        test_data = InferenceImageDataset(image_list=json_helper.model_input_retrieve(),transform=ToTensor())
        global_output_tensor = torch.tensor([])
        self.get_model()
        is_use_cuda = torch.cuda.is_available()

        self.model_instance.eval()

        with torch.no_grad():
            for input_batch in test_data:
                if input_batch==None:
                    break
                if is_use_cuda:
                    input_batch = input_batch.cuda()

                outputs = self.model_instance(input_batch)
                outputs = F.softmax(outputs, dim=1)

                global_output_tensor = torch.cat([global_output_tensor,torch.argmax(outputs,dim=1).cpu()],dim=0)
        
            final_output_classlist = global_output_tensor.cpu().tolist()
            return final_output_classlist


"""
End
"""


"""
Visualization Class : gives the final labelled image with bounding boxes
"""

class Visualization:
    def __init__(self,image_location):
        self.image_location = image_location
        self.image = cv2.imread(self.image_location)
        self.colors = {0 : (255,0,0), 1 : (0,0,255) , 2 : (0,255,0) , 3 : (128,0,128)}
    
    def visualize(self,temp_file):
        json_instance = JSONhelper(temp_file_path=temp_file)
        json_instance.generate_bbox_json_files(self.image_location,temp_file)
        model = Model()
        output_list = model.predict(temp_file_path=temp_file)
        bbox_info,rev_map = json_instance.retrieve_json()

        for cnt in range(len(output_list)):
            bb_dim = bbox_info[rev_map[str(cnt)]['file']][rev_map[str(cnt)]['index']]['bb_dim']
            cv2.rectangle(self.image,(bb_dim[0],bb_dim[1]),(bb_dim[2],bb_dim[3]),self.colors[output_list[cnt]],3)

        return self.image

"""
End
"""


def process_multiple_pages_TextualAttribute(folder_path,temp_file_path):
    json_helper = JSONhelper(temp_file_path=temp_file_path)
    json_helper.generate_bbox_json_files(folder_path,temp_file_path=temp_file_path)
    model = Model()
    output_list = model.predict(temp_file_path=temp_file_path)

    files = [join(folder_path, i) for i in os.listdir(folder_path)]
    doc = DocumentFile.from_images(files)
    a = PREDICTOR_V2(doc)
    ret = []
    cnt=0
    prev_cnt=0
    for idx in range(len(files)):
        page = a.pages[idx]
		# in the format (height, width)
        dim = page.dimensions
        lines = []
        for i in page.blocks:
            lines += i.lines
            regions = []
        map_int_to_str= {0:'none',1:'bold',2:'italic'}
        for i, line in enumerate(lines):
            for word in line.words:
                attb = {'none':False,'bold':False,'italic':False}
                if output_list[cnt]==3:
                    attb[map_int_to_str[1]]=True
                    attb[map_int_to_str[2]]=True
                else:
                    attb[map_int_to_str[output_list[cnt]]]=True

                regions.append(
					Region.from_bounding_box(convert_geometry_to_bbox(word.geometry, dim, padding=5),
                              attb=attb,
                              order=cnt-prev_cnt,
                              line=i+1)
				)
                cnt+=1
        prev_cnt = cnt

        ret.append(
			LayoutImageResponse(
				regions=regions.copy(),
				image_name=basename(files[idx])
			)
		)
    return ret