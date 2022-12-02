import os
import sys
import argparse
import shutil
import math
import base64
import io
from collections import OrderedDict
from multiprocessing import Pool
import json

import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import PIL.ExifTags
import PIL.Image
import PIL.ImageOps

class Labelme2YOLO(object):
    
    def __init__(self, json_dir,image_dir,classes=None):
        self._json_dir = json_dir
        self._image_dir = image_dir
        if classes is not None :
            self._label_id_map = classes
        else :
            self._label_id_map = self._get_label_id_map(self._json_dir)
        
    def _make_train_val_dir(self):
        self._label_dir_path = os.path.join(self._json_dir, 
                                            'YOLODataset/labels/')
        self._image_dir_path = os.path.join(self._json_dir, 
                                            'YOLODataset/images/')
        
        for yolo_path in (os.path.join(self._label_dir_path + 'train/'), 
                          os.path.join(self._label_dir_path + 'val/'),
                          os.path.join(self._label_dir_path + 'test/'),
                          os.path.join(self._image_dir_path + 'train/'), 
                          os.path.join(self._image_dir_path + 'val/'),
                          os.path.join(self._image_dir_path + 'test/')):
            if os.path.exists(yolo_path):
                shutil.rmtree(yolo_path)
            
            os.makedirs(yolo_path)    
                
    def _get_label_id_map(self, json_dir):
        label_set = set()
    
        for file_name in os.listdir(json_dir):
            if file_name.endswith('json'):
                json_path = os.path.join(json_dir, file_name)
                data = json.load(open(json_path))
                for shape in data['shapes']:
                    label_set.add(shape['label'])
        
        return OrderedDict([(label, label_id) \
                            for label_id, label in enumerate(label_set)])
    def _train_test_split(self,json_names,val_size=0.25, test_size=0.1):
        n = len(json_names)
        ntest = int(n*test_size)
        ntrainval = n-ntest
        nval = int(ntrainval*val_size)
        if test_size > 0 :
            trainval,test = train_test_split(json_names,test_size=test_size,random_state=42)
        else :
            trainval,test = json_names,[]
        if val_size > 0 :
            train,val = train_test_split(trainval,test_size=val_size,random_state=42)
        else :
             train,val = trainval,[]

        # move image to dir image 
        for img_dir,image in zip(('train/','val/','test/'),(train,val,test)):
            for img in image:
                src_path = os.path.join(self._image_dir, img.replace('.json', '.jpg'))
                dist_path = os.path.join(self._image_dir_path,img_dir, img.replace('.json', '.jpg'))
                shutil.copy(src_path,dist_path)

        return train,val,test
    
    def convert(self, val_size=0.25, test_size=0.1):
        json_names = [file_name for file_name in os.listdir(self._json_dir) \
                      if os.path.isfile(os.path.join(self._json_dir, file_name)) and \
                      file_name.endswith('.json')]
        folders =  [file_name for file_name in os.listdir(self._json_dir) \
                    if os.path.isdir(os.path.join(self._json_dir, file_name))]
        self._make_train_val_dir()
        train_json_names, val_json_names, test_json_names = self._train_test_split(json_names, val_size, test_size)
        print(train_json_names,val_json_names,test_json_names)
        # sys.exit()
        # convert labelme object to yolo format object, and save them to files
        # also get image from labelme json file and save them under images folder
        for target_dir, json_names in zip(('train/', 'val/', 'test/'), 
                                          (train_json_names, val_json_names, test_json_names)):
            # pool = Pool(os.cpu_count() - 1)
            for json_name in json_names:
                self.covert_json_to_text(os.path.join(self._label_dir_path,target_dir), json_name)
            # pool.close()
            # pool.join()
        
        print('Generating dataset.yaml file ...')
        self._save_dataset_yaml()

    def covert_json_to_text(self, target_dir, json_name):
        json_path = os.path.join(self._json_dir, json_name)
        json_data = json.load(open(json_path))
                
        print('Converting %s for %s ...' % (json_name, target_dir))
                    
        id_list,yolo_obj_list = self._get_yolo_object_list(json_data)

        self._save_yolo_label(json_name, 
                                      target_dir,id_list, 
                                      yolo_obj_list)
                
    def _get_yolo_object_list(self, json_data):
        yolo_obj_list = []
        id_list = []
        img_h, img_w = json_data["imageHeight"],json_data["imageWidth"]
        for shape in json_data['shapes']:
            # labelme circle shape is different from others
            # it only has 2 points, 1st is circle center, 2nd is drag end point
            if shape['shape_type'] == 'circle':
                id,yolo_obj = self._get_circle_shape_yolo_object(shape, img_h, img_w)
            else:
                id,yolo_obj = self._get_other_shape_yolo_object(shape, img_h, img_w)
            id_list.append(id)
            yolo_obj_list.append(yolo_obj)
            
        return id_list, yolo_obj_list
    
    def _get_circle_shape_yolo_object(self, shape, img_h, img_w):
        obj_center_x, obj_center_y = shape['points'][0]
        
        radius = math.sqrt((obj_center_x - shape['points'][1][0]) ** 2 + 
                           (obj_center_y - shape['points'][1][1]) ** 2)
        obj_w = 2 * radius
        obj_h = 2 * radius
        
        yolo_center_x= round(float(obj_center_x / img_w), 6)
        yolo_center_y = round(float(obj_center_y / img_h), 6)
        yolo_w = round(float(obj_w / img_w), 6)
        yolo_h = round(float(obj_h / img_h), 6)
            
        label_id = self._label_id_map[shape['label']]
        
        return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h
    
    def _get_other_shape_yolo_object(self, shape, img_h, img_w):
        
        points = []
        for xy in shape["points"]:
            x = round(float(xy[0] / img_w), 6)
            y = round(float(xy[1] / img_h), 6)
            points.append(x)
            points.append(y)

        label_id = self._label_id_map[shape['label']]
        
        # return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h
        return label_id, points
    
    def _save_yolo_label(self, json_name, label_dir_path, id_list,yolo_obj_list):
        txt_path = os.path.join(label_dir_path, json_name.replace('.json', '.txt'))

        with open(txt_path, 'w+') as f:
            for yolo_obj_idx, yolo_obj in zip(id_list, yolo_obj_list):
                yolo_obj_line = str(yolo_obj_idx) + " " + " ".join([("%.6f" % a) for a in yolo_obj]) + "\n" \
                        if len(id_list) > 1 else \
                            str(yolo_obj_idx) + " " + " ".join([("%.6f" % a) for a in yolo_obj])
                f.write(yolo_obj_line)
                
    def _save_dataset_yaml(self):
        yaml_path = os.path.join(self._json_dir, 'YOLODataset/', 'dataset.yaml')
        
        with open(yaml_path, 'w+') as yaml_file:
            yaml_file.write('train: %s\n' % \
                            os.path.join(self._image_dir_path, 'train/'))
            yaml_file.write('val: %s\n\n' % \
                            os.path.join(self._image_dir_path, 'val/'))
            yaml_file.write('test: %s\n\n' % \
                            os.path.join(self._image_dir_path, 'test/'))
            yaml_file.write('nc: %i\n\n' % len(self._label_id_map))
            
            names_str = ''
            for label, _ in self._label_id_map.items():
                names_str += "'%s', " % label
            names_str = names_str.rstrip(', ')
            yaml_file.write('names: [%s]' % names_str)
            
classes = {"defect":0,"other":1}
json_dirpath = "datasets/knuckle/"
image_dirpath = "datasets/knuckle/"
L2Y = Labelme2YOLO(json_dirpath,image_dirpath)
L2Y.convert(val_size = 0.15,test_size = 0.0)