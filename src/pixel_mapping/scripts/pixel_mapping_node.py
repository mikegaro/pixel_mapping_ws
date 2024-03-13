#!/usr/bin/python3

#ROS
import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from unet.msg import Inference
from cv_bridge import CvBridge
import math

import numpy as np
import cv2
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
from scipy.optimize import minimize

class PixelMappingNode():
    def __init__(self):
        #ROS Objects
        self.detection_subscriber = rospy.Subscriber("/unet/pixel_detection", Inference, self.callback, queue_size=1)
        self.points_publisher = rospy.Publisher("/pixel_map/points", Float32MultiArray, queue_size=1)
        self.image_publisher = rospy.Publisher("/pixel_map/image", Image, queue_size=1)
        self.br = CvBridge()

        #CAMERA PARAMETERS
        self.focal_length_mm        = 50
        self.sensor_height_mm       = 8.33
        self.camera_height_m        = 1.955
        self.sensor_width_mm        = 14.6
        self.image_resolution_height= 720
        self.image_resolution_width = 1280

        # Calculated fields
        self.degrees_per_pixel_vertical = 2 * math.atan((self.sensor_height_mm / 2) / self.focal_length_mm) * (180 / math.pi) / self.image_resolution_height
        self.degrees_per_pixel_horizontal = 2 * math.atan((self.sensor_width_mm / 2) / self.focal_length_mm) * (180 / math.pi) / self.image_resolution_width
        self.midpoint_x_location = 231.30 + 436.166
        self.calibrated_tilt_angle = self.calibrate_tilt_angle()

    def callback(self, msg):
        pixel_list = list(zip(msg.x, msg.y))
        print(pixel_list[0])
        print(pixel_list[-1])
        sorted_pixel_list = sorted(pixel_list, key=lambda a: a[1])
        # for i in sorted_pixel_list:
        #     print(i)
        # #print(f"{sorted_pixel_list[0]}, {sorted_pixel_list[-1]}")
        mean_point_dict = {}
        for x, y in sorted_pixel_list:
            x_pixel = mean_point_dict.get(y, [])
            x_pixel.append(x)
            mean_point_dict[y] = x_pixel

        # for y in mean_point_dict.keys():
        #     print(f"{y} is {mean_point_dict[y]}")

        virtual_middle_rail = {}
        for y in mean_point_dict.keys():
            mean_pixel_x = int(np.average(mean_point_dict[y]))
            #print(f"In {y} the ¨{mean_point_dict[y]} average is {mean_pixel_x}")
            virtual_middle_rail[y] = mean_pixel_x

        virtual_middle_rail_list = list(virtual_middle_rail.items())
        
        x_array = np.array([x for _, x in virtual_middle_rail_list])
        y_array = np.array([y for y, _ in virtual_middle_rail_list])

        x_y_smooth_fit = make_interp_spline(y_array, x_array)
        x_array_smooth = x_y_smooth_fit(y_array)
        x_array_filtered = savgol_filter(x_array, 51, 2)
        image = np.array(self.br.imgmsg_to_cv2(msg.image))

        for x,y in list(zip(x_array_filtered[::50], y_array[::50])):
            #print(f"In {y} the average is {x}")
            vertical_distance = self.calculate_ground_distance_from_bottom(720-y, self.calibrated_tilt_angle)
            print(f"y = {y} -> distance = {vertical_distance}")
            cv2.circle(image, (int(x),int(y)), radius=5, color=(0, 0, 255), thickness=-1)
        #image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.image_publisher.publish(self.br.cv2_to_imgmsg(image, encoding='rgb8'))

    def calibrate_tilt_angle(self):
        # Calibration logic for the camera tilt angle
        pixels_marks = [691.1904761904761,
                        932.0229562398083,
                        1064.2998080804832,
                        1134.5479913753973,
                        1147.7669879116024,
                        1155.5978616716056,
                        1166.3117464334957]
        distances = [25, 
                     50, 
                     100, 
                     200, 
                     250, 
                     300, 
                     400]

        def total_error(tilt_angle):
            return sum(abs(self.calculate_ground_distance_from_bottom(pix, tilt_angle) - dist)
                       for pix, dist in zip(pixels_marks, distances))

        result = minimize(total_error, -5, method='Nelder-Mead')
        return result.x[0]

    def calculate_ground_distance_from_bottom(self, pixels_from_bottom, tilt_angle_deg):
        angle_from_bottom_deg = (self.image_resolution_height - pixels_from_bottom) * self.degrees_per_pixel_vertical
        total_angle_deg = angle_from_bottom_deg + tilt_angle_deg
        return self.camera_height_m / math.tan(math.radians(total_angle_deg))

    def calculate_hypotenuse_pixels(self, pixels_x, pixels_y):
        return math.sqrt(pixels_x**2 + pixels_y**2)

def main():
    rospy.init_node("pixel_mapping_node")
    node = PixelMappingNode()
    rospy.spin()
    pass

if __name__ == "__main__":
    exit(main())