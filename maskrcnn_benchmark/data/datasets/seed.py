from maskrcnn_benchmark.structures.bounding_box import BoxList
# from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

from PIL import Image
# import cv2
import os
import pandas as pd
import torch
import torch.utils.data
import numpy as np
import cv2
import xml.etree.ElementTree
from .get_bb  import *
class SeedDataset(torch.utils.data.Dataset):
    CLASSES_STRAT1 = (
        "__background__ ",
        "germinated",
        "non-germinated"
        # "seed",
        # "radical"
    )
    CLASSES_STRAT2 = (
        "__background__ ",
        # "germinated",
        # "non-germinated"
        "seed",
        "radical"
    )

    def __init__(self, root_dir, split="", strategy = 1, transforms=None):
        # as you would do normally
        self.split = split
        self.nb_data = 127
        if ("train".upper() in self.split.upper()):
            self.starting_sample = 1
            self.number_sample = int( np.floor(self.nb_data * 0.8) )
        # elif ("val".upper() in self.split.upper()):
        #     self.starting_sample = int( np.floor(self.nb_data * 0.6) )
        #     self.number_sample = int(np.floor(self.nb_data * 0.2))
        elif ("test".upper() in self.split.upper()):
            self.starting_sample = int(np.floor(self.nb_data * 0.8))  + 1
            # self.number_sample = int(np.floor(self.nb_data * 0.2))
            self.number_sample = self.nb_data - self.starting_sample + 1
        else:
            self.starting_sample = 1
            self.number_sample = self.nb_data

        self.root_dir = root_dir
        self.strategy = strategy
        self.transforms = transforms
        if strategy == 1:    
            cls = SeedDataset.CLASSES_STRAT1
        else:
             cls = SeedDataset.CLASSES_STRAT2
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.b_boxes, self.labels = self.get_ground_bb()

    def get_ground_bb(self):
        boxes = []
        labels = []
        if self.strategy == 2:
            xml_names_seed = [f'/home/ramazam/Documents/Spring 2019/CV/seeds_proj/seeds_2/image{i:03d}_s.xml' for i in range(1, self.nb_data + 1)]
            xml_names_rad = [f'/home/ramazam/Documents/Spring 2019/CV/seeds_proj/seeds_2/image{i:03d}.xml' for i in range(1, self.nb_data + 1)]
            for (xml_1, xml_2) in zip(xml_names_seed, xml_names_rad):
                ground_boxes, ground_labels = get_ground_bb_2( xml_1, xml_2)
                boxes.append(ground_boxes)
                labels.append(ground_labels )
        else:
            xml_names = [f'/home/ramazam/Documents/Spring 2019/CV/seeds_proj/seeds/image{i:03d}.xml' for i in range(1, self.nb_data + 1)]            
            for xml_1 in xml_names:
                ground_boxes, ground_labels = get_ground_bb( xml_1)
                boxes.append(ground_boxes)
                labels.append(ground_labels )
        return boxes, labels

    def __getitem__(self, idx):       
        num = idx  + self.starting_sample
        # print(num)
        # labels = []
        boxes = self.b_boxes[num - 1]
        labels_xml = self.labels[num - 1]
        if self.strategy == 1:
            image_name = f'/home/ramazam/Documents/Spring 2019/CV/seeds_proj/seeds/image{num:03d}.jpg'
            labels = [1 if b == 'germinated' else 2 for b in  labels_xml]
            
        else:
            image_name = f'/home/ramazam/Documents/Spring 2019/CV/seeds_proj/seeds_2/image{num:03d}.jpg'          
            labels = [1 if b == 'seed' else 2 for b in  labels_xml]
            # print(labels)

        # print(labels)
        image = Image.open( image_name).convert('RGB')
        difficult = np.zeros(len(boxes))

        labels = torch.tensor(labels)
        difficult = torch.tensor(difficult)
        # print(labels)
        # create a BoxList from the boxes
        boxlist = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        # boxlist.add_field("masks", masks)
        boxlist.add_field("labels", labels)
        boxlist.add_field("difficult", difficult)

        if self.transforms:
            image, boxlist = self.transforms(image, boxlist)
        # return the image, the boxlist and the idx in your dataset
        return image, boxlist, idx

    def __len__(self):
        return self.number_sample

    def get_groundtruth(self, index):
        _, boxlist, _ = self[index]                          
        return boxlist

    def get_strategy(self):
        return self.strategy
        
    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        num  = idx + self.starting_sample
        if self.strategy == 1:
            image_name = f'/home/ramazam/Documents/Spring 2019/CV/seeds_proj/seeds/image{num:03d}.jpg'
        else:
            image_name = f'/home/ramazam/Documents/Spring 2019/CV/seeds_proj/seeds_2/image{num:03d}.jpg'
        image = Image.open(image_name).convert('RGB')
        return {"height": image.size[1], "width": image.size[0], "name": image_name, "id": num}

    def map_class_id_to_class_name(self, class_id):
        if self.strategy == 1:    
             return SeedDataset.CLASSES_STRAT1[class_id]
        return SeedDataset.CLASSES_STRAT2[class_id]
#python ./tools/train_net.py --config-file /home/ramazam/Documents/maskrcnn-benchmark/configs/seed/e2e_faster_rcnn_R_50_C4_1x_seed.yamlfrom maskrcnn_benchmark.structures.bounding_box import BoxList
# from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

# from PIL import Image
# # import cv2
# import os
# import pandas as pd
# import torch
# import torch.utils.data
# import numpy as np
# import cv2
# import xml.etree.ElementTree
# from .get_bb  import *
# class SeedDataset(torch.utils.data.Dataset):
#     CLASSES_STRAT1 = (
#         "__background__ ",
#         "germinated",
#         "non-germinated"
#         # "seed",
#         # "radical"
#     )
#     CLASSES_STRAT2 = (
#         "__background__ ",
#         # "germinated",
#         # "non-germinated"
#         "seed",
#         "radical"
#     )

#     def __init__(self, root_dir, split="", strategy = 1, transforms=None):
#         # as you would do normally
#         self.split = split
#         self.nb_data = 127
#         if ("train".upper() in self.split.upper()):
#             self.starting_sample = 1
#             self.number_sample = int( np.floor(self.nb_data * 0.8) )
#         # elif ("val".upper() in self.split.upper()):
#         #     self.starting_sample = int( np.floor(self.nb_data * 0.6) )
#         #     self.number_sample = int(np.floor(self.nb_data * 0.2))
#         elif ("test".upper() in self.split.upper()):
#             self.starting_sample = int(np.floor(self.nb_data * 0.8))  + 1
#             # self.number_sample = int(np.floor(self.nb_data * 0.2))
#             self.number_sample = self.nb_data - self.starting_sample + 1
#         else:
#             self.starting_sample = 1
#             self.number_sample = self.nb_data

#         self.root_dir = root_dir
#         self.strategy = strategy
#         self.transforms = transforms
#         if strategy == 1:    
#             cls = SeedDataset.CLASSES_STRAT1
#         else:
#              cls = SeedDataset.CLASSES_STRAT2
#         self.class_to_ind = dict(zip(cls, range(len(cls))))
#         self.b_boxes, self.labels = self.get_ground_bb()

#     def get_ground_bb(self):
#         boxes = []
#         labels = []
#         if self.strategy == 2:
#             xml_names_seed = [f'/home/ramazam/Documents/Spring 2019/CV/seeds_proj/seeds_2/image{i:03d}_s.xml' for i in range(1, self.nb_data + 1)]
#             xml_names_rad = [f'/home/ramazam/Documents/Spring 2019/CV/seeds_proj/seeds_2/image{i:03d}.xml' for i in range(1, self.nb_data + 1)]
#             for (xml_1, xml_2) in zip(xml_names_seed, xml_names_rad):
#                 ground_boxes, ground_labels = get_ground_bb_2( xml_1, xml_2)
#                 boxes.append(ground_boxes)
#                 labels.append(ground_labels )
#         else:
#             xml_names = [f'/home/ramazam/Documents/Spring 2019/CV/seeds_proj/seeds/image{i:03d}.xml' for i in range(1, self.nb_data + 1)]            
#             for xml_1 in xml_names:
#                 ground_boxes, ground_labels = get_ground_bb( xml_1)
#                 boxes.append(ground_boxes)
#                 labels.append(ground_labels )
#         return boxes, labels

#     def __getitem__(self, idx):       
#         num = idx  + self.starting_sample
#         print(num)
#         # labels = []
#         boxes = self.b_boxes[num - 1]
#         labels_xml = self.labels[num - 1]
#         if self.strategy == 1:
#             image_name = f'/home/ramazam/Documents/Spring 2019/CV/seeds_proj/seeds/image{num:03d}.jpg'
#             labels = [1 if b == 'germinated' else 2 for b in  labels_xml]
            
#         else:
#             image_name = f'/home/ramazam/Documents/Spring 2019/CV/seeds_proj/seeds_2/image{num:03d}.jpg'          
#             labels = [1 if b == 'seed' else 2 for b in  labels_xml]
#             # print(labels)

#         # print(labels)
#         image = Image.open( image_name).convert('RGB')
#         difficult = np.zeros(len(boxes))

#         labels = torch.tensor(labels)
#         difficult = torch.tensor(difficult)
#         # print(labels)
#         # create a BoxList from the boxes
#         boxlist = BoxList(boxes, image.size, mode="xyxy")
#         # add the labels to the boxlist
#         # boxlist.add_field("masks", masks)
#         boxlist.add_field("labels", labels)
#         boxlist.add_field("difficult", difficult)

#         if self.transforms:
#             image, boxlist = self.transforms(image, boxlist)
#         # return the image, the boxlist and the idx in your dataset
#         return image, boxlist, idx

#     def __len__(self):
#         return self.number_sample

#     def get_groundtruth(self, index):
#         _, boxlist, _ = self[index]                          
#         return boxlist

#     def get_strategy(self):
#         return self.strategy
        
#     def get_img_info(self, idx):
#         # get img_height and img_width. This is used if
#         # we want to split the batches according to the aspect ratio
#         # of the image, as it can be more efficient than loading the
#         # image from disk
#         num  = idx + self.starting_sample
#         if self.strategy == 1:
#             image_name = f'/home/ramazam/Documents/Spring 2019/CV/seeds_proj/seeds/image{num:03d}.jpg'
#         else:
#             image_name = f'/home/ramazam/Documents/Spring 2019/CV/seeds_proj/seeds_2/image{num:03d}.jpg'
#         image = Image.open(image_name).convert('RGB')
#         return {"height": image.size[1], "width": image.size[0], "name": image_name, "id": num}

#     def map_class_id_to_class_name(self, class_id):
#         if self.strategy == 1:    
#              return SeedDataset.CLASSES_STRAT1[class_id]
#         return SeedDataset.CLASSES_STRAT2[class_id]
# #python ./tools/train_net.py --config-file /home/ramazam/Documents/maskrcnn-benchmark/configs/seed/e2e_faster_rcnn_R_50_C4_1x_seed.yaml