#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class VideoNode():
  def __init__(self, image_file:str, fps=15):
    self.video_file = image_file
    self.pub = rospy.Publisher("/camera/image", Image, queue_size=1)
    self.br = CvBridge()
    self.rate = rospy.Rate(fps)
    self.image = cv2.imread(image_file)

  def publish_frame(self):
        self.pub.publish(self.br.cv2_to_imgmsg(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), 
                                                 encoding='rgb8'))

def main():
  rospy.init_node("video_publisher_node")
  image_file = rospy.get_param("image_file")
  if not os.path.isfile(image_file):
    raise Exception(f"Video file not found: {image_file}")
  
  node = VideoNode(image_file)
  try:
    while not rospy.is_shutdown():
        node.publish_frame()
        node.rate.sleep()
  except rospy.ROSInterruptException:
    pass

if __name__ == '__main__':
  exit(main())