#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class VideoNode():
  def __init__(self, video_file:str):
    self.video_file = video_file
    self.pub = rospy.Publisher("/camera/image", Image, queue_size=1)
    self.cap = cv2.VideoCapture(video_file)
    self.br = CvBridge()
    self.fps = self.cap.get(cv2.CAP_PROP_FPS)
    self.rate = rospy.Rate(self.fps)

  def publish_frame(self):
    while not rospy.is_shutdown():
        ret, frame = self.cap.read()

        if ret:
          self.pub.publish(self.br.cv2_to_imgmsg(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                                 encoding='rgb8'))
          self.rate.sleep()
        else:
          self.cap.release()
          self.cap = cv2.VideoCapture(self.video_file)

def main():
  rospy.init_node("video_publisher_node")
  video_file = rospy.get_param("video_file")
  if not os.path.isfile(video_file):
    raise Exception(f"Video file not found: {video_file}")
  
  node = VideoNode(video_file)
  try:
    node.publish_frame()
  except rospy.ROSInterruptException:
    pass

if __name__ == '__main__':
  exit(main())