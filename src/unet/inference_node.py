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
        cv2_img = adjust_vibrance(1.5, cv2_img_original)
        #cv2_img = adjust_contrast(cv2_img_original)
        #cv2_img = cv2.GaussianBlur(cv2_img, (5,5),cv2.BORDER_DEFAULT)
        cv2_img = increase_sharpening(cv2_img_original)
        image = PIL.Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
        img_transforms = tf.Compose([
            tf.Resize((HEIGHT, WIDTH)),
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

        annotated = np.copy(cv2.resize(cv2_img_original, (WIDTH, HEIGHT), interpolation = cv2.INTER_AREA))
        
        idxs = np.where(prediction == 1.0)
        x_list = idxs[2][:]
        y_list = idxs[1][:]

        for i in range(0,len(idxs[1])-1):
            annotated[idxs[1][i]][idxs[2][i]] = [0,255,0] # Green
        
        inference_msg = Inference()
        inference_msg.x = x_list
        inference_msg.y = y_list
        inference_msg.length = len(idxs[1])
        if(len(idxs[1]) == len(idxs[2])):
            print("Publishing inference")
            inference_msg.image = self.br.cv2_to_imgmsg(cv2.resize(cv2_img_original, (WIDTH, HEIGHT)), encoding='rgb8')
        
        self.pixel_pub.publish(inference_msg)
        self.annotated_image_pub.publish(self.br.cv2_to_imgmsg(annotated, encoding='rgb8'))

        
def main():
    rospy.init_node("unet_inference_node")
    model_file = rospy.get_param("model_file")
    node = UnetNode(model_file)
    rospy.spin()


if __name__ == "__main__":
    exit(main())

