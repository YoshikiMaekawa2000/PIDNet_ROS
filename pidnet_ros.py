#!/usr/bin/python3
import rospy
import os
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image as ImageMsg
from cv_bridge import CvBridge

import cv2
import numpy as np
from PIDNet.models import pidnet
import torch
import torch.nn.functional as F
from PIL import Image

color_map = [(128, 64,128),
             (244, 35,232),
             ( 70, 70, 70),
             (102,102,156),
             (190,153,153),
             (153,153,153),
             (250,170, 30),
             (220,220,  0),
             (107,142, 35),
             (152,251,152),
             ( 70,130,180),
             (220, 20, 60),
             (255,  0,  0),
             (  0,  0,142),
             (  0,  0, 70),
             (  0, 60,100),
             (  0, 80,100),
             (  0,  0,230),
             (119, 11, 32)]


class PIDNet_ROS:
    def __init__(self):

        self.node_name =  "pidnet_ros"
        rospy.init_node(self.node_name)

        weight_path = "/PIDNet/pretrained_models/cityscapes/PIDNet_S_Cityscapes_test.pt"
        predict_model = pidnet.get_pred_model('pidnet-s', 19)

        self.predictor = self.load_pretrained(predict_model, os.path.join(os.getcwd() + weight_path))
        self.predictor.eval()
        self.bridge = CvBridge()

        self.mean = rospy.get_param('mean', [0.485, 0.456, 0.406])
        self.std = rospy.get_param('std', [0.229, 0.224, 0.225])
        self.use_speed_ver = rospy.get_param('use_speed_ver', False)

        self.subscribed_img = False
        self.image_sub = rospy.Subscriber('/CompressedImage', CompressedImage, self.image_callback)
        self.pub = rospy.Publisher('/pidnet', ImageMsg, queue_size=10)
        rospy.Timer(rospy.Duration(0.05), self.timerCallback)

    def timerCallback(self, event):
        if self.subscribed_img:
            with torch.no_grad():
                pred = self.predictor(self.img)
                pred = F.interpolate(pred, size=self.img.size()[-2:],
                        mode='bilinear', align_corners=True)
                pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
                for i, color in enumerate(color_map):
                    for j in range(3):
                        self.sv_img[:,:,j][pred==i] = color_map[i][j]

                sv_img = cv2.cvtColor(self.sv_img, cv2.COLOR_BGR2RGB)

                result_msg = self.bridge.cv2_to_imgmsg(sv_img, encoding="passthrough")

                self.pub.publish(result_msg)

    def image_callback(self, msg):
        with torch.no_grad():
            img = self.bridge.compressed_imgmsg_to_cv2(msg)

            self.sv_img = np.zeros_like(img).astype(np.uint8)
            img = self.input_transform(img)
            img = img.transpose((2, 0, 1)).copy()
            img = torch.from_numpy(img).unsqueeze(0)
            self.img = img
        self.subscribed_img = True

    def input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def load_pretrained(self, model, pretrained):
        pretrained_dict = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
        msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
        print('Attention!!!')
        print(msg)
        print('Over!!!')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict = False)

        return model

if __name__=="__main__":
    PIDNet_ROS()
    rospy.spin()

