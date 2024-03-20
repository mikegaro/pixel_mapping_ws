#!/usr/bin/python3

#Pytorch
import torch
import torchvision.transforms as tf
from model.model import UNet
from model.utils import *

#ROS
import rospy
from sensor_msgs.msg import Image
from unet.msg import Inference
from cv_bridge import CvBridge

#UTILS
import cv2
import PIL
import numpy as np

WIDTH = 1920
HEIGHT = 1080
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class UnetNode:
    def __init__(self, model_file):
        #Init network
        self.net = UNet()
        self.net.to(DEVICE)
        self.net.load_state_dict(torch.load(model_file, map_location='cpu'))
        self.net.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True

        self.annotated_image_pub = rospy.Publisher("unet/annotated_image", Image, queue_size=1)
        self.camera_image_sub = rospy.Subscriber("camera/image", Image, self.callback ,queue_size=1)
        self.pixel_pub = rospy.Publisher("unet/pixel_detection", Inference, queue_size=1)
        self.br = CvBridge()

    def callback(self, msg):
        cv2_img_original = self.br.imgmsg_to_cv2(msg)
        cv2_img_original = cv2.resize(cv2_img_original, (3840, 2160), interpolation = cv2.INTER_AREA)
        height, width, channel = cv2_img_original.shape

        h_split_index = height//2
        w_split_index = width//2
        print(f"({height}, {width})")
        
        top_left_image = cv2_img_original[:h_split_index,       :w_split_index]
        top_right_image = cv2_img_original[:h_split_index,      w_split_index:]
        bottom_left_image = cv2_img_original[h_split_index:,    :w_split_index]
        bottom_right_image = cv2_img_original[h_split_index:,   w_split_index:]

        imgs = [top_left_image, top_right_image, bottom_left_image, bottom_right_image]
        imgs_new =[]

        for i in imgs:
            h, w, _ = i.shape
            cv2_img = i
            print(f"img shape: {w}, {h}")
            cv2_img = adjust_vibrance(1.5, cv2_img)
            #cv2_img = adjust_contrast(cv2_img)
            #cv2_img = cv2.GaussianBlur(cv2_img, (5,5),cv2.BORDER_DEFAULT)
            #cv2_img = increase_sharpening(cv2_img)

            image = PIL.Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
            img_transforms = tf.Compose([
                tf.ToTensor(),
                tf.Normalize(mean=[0.2752, 0.2546, 0.2860],
                             std=[0.1541, 0.1535, 0.1679])
                ])
            image = img_transforms(image).unsqueeze(0)
            image = image.to(DEVICE)
        
            prediction=None
            with torch.no_grad():
                prediction = self.net(image)

            prediction = torch.sigmoid(prediction)
            prediction = prediction.squeeze(1).permute(0,1,2).to("cpu")
            prediction = prediction.detach().numpy()
            prediction[prediction > 0.5] = 1.0
            prediction[prediction <= 0.5] = 0.0
            annotated = np.copy(i)
            idxs = np.where(prediction == 1.0)
            x_list = idxs[2][:]
            y_list = idxs[1][:]
            for i in range(0,len(idxs[1])-1):
                annotated[idxs[1][i]][idxs[2][i]] = [0,255,0] # Green
            
            imgs_new.append(annotated)

        img_top = cv2.hconcat([imgs_new[0], imgs_new[1]])
        img_bottom = cv2.hconcat([imgs_new[2], imgs_new[3]])

        reconstructed_image = cv2.vconcat([img_top,img_bottom])
        h, w, c = reconstructed_image.shape
        print(f"Reconstructed ({h},{w})")
        #reconstructed_image = cv2.resize(reconstructed_image, (1080, 1920), interpolation = cv2.INTER_AREA)        
        self.annotated_image_pub.publish(self.br.cv2_to_imgmsg(reconstructed_image, encoding='rgb8'))

        
def main():
    rospy.init_node("unet_inference_node")
    model_file = rospy.get_param("model_file")
    node = UnetNode(model_file)
    rospy.spin()


if __name__ == "__main__":
    exit(main())