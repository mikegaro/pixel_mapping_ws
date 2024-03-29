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

        # LAT LONG PARAMETERS 
        self.GPSposition_lat = 42.16179665 # Example usage location
        self.GPSposition_long = -79.96923472 # Example usage location

        # Orientation angle (Northeast = 55 degrees)
        self.orientation_angle = 55

        # Track width 
        self.track_width = 1.435 # Value obtained with Wabtec

        # Calculated fields
        self.degrees_per_pixel_vertical = 2 * math.atan((self.sensor_height_mm / 2) / self.focal_length_mm) * (180 / math.pi) / self.image_resolution_height
        self.degrees_per_pixel_horizontal = 2 * math.atan((self.sensor_width_mm / 2) / self.focal_length_mm) * (180 / math.pi) / self.image_resolution_width
        #self.midpoint_x_location = 231.30 + 436.166
        self.calibrated_tilt_angle = self.calibrate_tilt_angle()

    def callback(self, msg):
        #midpoint_x_reference = 501.64  # The calculated midpoint x-coordinate for 1280x720 resolution

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

        # Determine the bottom most y-coordinate
        max_y = max(mean_point_dict.keys())

        # Extract x-coordinates corresponding to the bottom-most y-coordinate 
        bottom_most_x_coords = mean_point_dict[max_y]

        # Calculate the midpoint of the x-coordinates at the bottom-most level
        if bottom_most_x_coords:
            min_x = min(bottom_most_x_coords)
            max_x = max(bottom_most_x_coords)
            # Calculate midpoint as the middle of the min and max x-coordinates. 
            midpoint_x_bottom = (min_x + max_x) / 2
            #print(f"Midpoint X-coordinate at the bottom-most y-level ({max_y}): {midpoint_x_bottom}")
        else: 
            print("No x-coordinates found at the bottom-most y-level.")
            midpoint_x_bottom = None  # Or handle this case as needed

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
            #print(f"y = {y} -> distance = {vertical_distance}")

            ### Horizontal calculation ###

            # Obtain real degree from vertical distance
            horizontal_real_distance_degree = self.degrees_per_pixel_horizontal * vertical_distance
            #print(horizontal_real_distance_degree)

            # Horizontal distance with no consideration of meters per pixels factor
            horizontal_distance_no_scaling = math.tan(math.radians(horizontal_real_distance_degree)) * vertical_distance

            # Offset from one of the railways to the middle. 
            horizontal_offset_pixels = min_x - midpoint_x_bottom

            # Divide the track width by half
            half_track_width_meters = self.track_width / 2

            # Now calculate the pixels per meter ratio using the half width
            if half_track_width_meters != 0:  # Prevent division by zero
                pixel_per_meter = horizontal_offset_pixels / half_track_width_meters
            else:
                pixel_per_meter = None  # Handle the case where the track half-width is zero or unknown
                print("Half track width in meters is not defined.")

            horizontal_distance = horizontal_distance_no_scaling / pixel_per_meter

            # Output the pixel per meter for further use
            #print(f"Pixel per meter: {pixel_per_meter}")

            # Adjusted horizontal calculation using the midpoint as reference
            #horizontal_offset = x - midpoint_x_bottom
            #print(horizontal_offset)
            #horizontal_angle = horizontal_offset * self.degrees_per_pixel_horizontal
            #horizontal_distance = vertical_distance * math.tan(math.radians(horizontal_angle))

            print(f"x = {horizontal_distance:.5f} m, y = {vertical_distance:.5f} m")

             # Convert to GPS coordinates
            new_lat, new_long = self.convert_to_gps(horizontal_distance, vertical_distance)
            #print(f"New GPS coordinates Latitude and Longitude: {new_lat}, {new_long}")
            #print(new_lat, new_long)

            cv2.circle(image, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1)
            text = f"{vertical_distance:.2f} m"
            cv2.putText(image, text, (int(x) + 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
           
        #image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if midpoint_x_bottom is not None:
            start_point = (int(midpoint_x_bottom), 0)  # Start at the top of the image
            end_point = (int(midpoint_x_bottom), image.shape[0])  # End at the bottom of the image
            color = (255, 0, 0)  # Red color in BGR
            thickness = 2  # Line thickness

            cv2.line(image, start_point, end_point, color, thickness)

        self.image_publisher.publish(self.br.cv2_to_imgmsg(image, encoding='rgb8'))


    def convert_to_gps(self, x_distance, y_distance):
        meters_per_degree_lat = 111139
        delta_lat = y_distance / meters_per_degree_lat
        meters_per_degree_lon = meters_per_degree_lat * math.cos(math.radians(self.GPSposition_lat))
        delta_long = x_distance / meters_per_degree_lon

        # Adjust delta_lat and delta_long according to the orientation angle
        adjusted_delta_lat, adjusted_delta_long = self.adjust_for_orientation(delta_lat, delta_long, self.orientation_angle)

        new_latitude = self.GPSposition_lat + adjusted_delta_lat
        new_longitude = self.GPSposition_long + adjusted_delta_long
        return new_latitude, new_longitude
    
    def adjust_for_orientation(self, delta_lat, delta_long, orientation_degrees):
        # Convert latitude and longitude deltas to meters
        lat_meters = delta_lat * 111139
        long_meters = delta_long * 111139  # This is an approximation, assuming the delta is small

        # Rotate the deltas by the orientation angle
        angle_rad = math.radians(orientation_degrees)
        adjusted_lat_meters = lat_meters * math.cos(angle_rad) - long_meters * math.sin(angle_rad)
        adjusted_long_meters = lat_meters * math.sin(angle_rad) + long_meters * math.cos(angle_rad)

        # Convert meters back to latitude and longitude deltas
        adjusted_delta_lat = adjusted_lat_meters / 111139
        adjusted_delta_long = adjusted_long_meters / 111139  # Again, an approximation

        return adjusted_delta_lat, adjusted_delta_long
    
    def calibrate_tilt_angle(self):
        # Calibration logic for the camera tilt angle
        pixels_marks = [452.78740157480314,
                        452.78740157480314 + 140.78837222960843,
                        452.78740157480314 + 140.78837222960843 + 70.04301537827146,
                        452.78740157480314 + 140.78837222960843 + 70.04301537827146 + 35.39672048977434,
                        452.78740157480314 + 140.78837222960843 + 70.04301537827146 + 35.39672048977434 + 2.923012983361732,
                        452.78740157480314 + 140.78837222960843 + 70.04301537827146 + 35.39672048977434 + 2.923012983361732 + 2.508970042590426,
                        452.78740157480314 + 140.78837222960843 + 70.04301537827146 + 35.39672048977434 + 2.923012983361732 + 2.508970042590426 + 0.260651629072697]
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

        result = minimize(total_error, -3, method='Nelder-Mead')
        return result.x[0]

    def calculate_ground_distance_from_bottom(self, pixels_from_bottom, tilt_angle_deg):
        # This is the vertical angle from the camera's optical axis to the pixel
        angle_from_bottom_deg = (self.image_resolution_height - pixels_from_bottom) * self.degrees_per_pixel_vertical
        
        # Adjusting for camera tilt: If the camera is tilted down, the tilt angle will be positive
        total_angle_deg = angle_from_bottom_deg + tilt_angle_deg
        
        # Ensuring the ground distance is calculated correctly
        ground_distance = self.camera_height_m / math.tan(math.radians(total_angle_deg))

        return max(ground_distance, 0)  # Ensures non-negative values

    def calculate_hypotenuse_pixels(self, pixels_x, pixels_y):
        return math.sqrt(pixels_x**2 + pixels_y**2)

def main():
    rospy.init_node("pixel_mapping_node")
    node = PixelMappingNode()
    rospy.spin()
    pass

if __name__ == "__main__":
    exit(main())