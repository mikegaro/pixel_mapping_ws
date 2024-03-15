#!/usr/bin/env /usr/bin/python3
import cv2
import numpy as np
from scipy.optimize import curve_fit

# ROS dependencies
import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

display_width = 1280
display_height = 720

# Define the size for the edge detection window 
edge_window_width = 540
edge_window_height = 360

# Initialize variables for calculating average brightness
total_brightness = 0
frame_count = 0 

# Initialize a list to store coordinates before starting the frame processing loo
rails_coordinates = []

# Threshold to detect curves
curve_threshold = 0.01

# Define the pixel-to-meter ratio
pixel_to_meter_ratio = 0.02  # Example value, adjust based on your calibration

fps = 20

class EdgeDetectionNode():
    def __init__(self, topic):
        self.sub = rospy.Subscriber(topic, Image, self.callback, queue_size=1)
        self.bridge = CvBridge()
        self.pub = rospy.Publisher("edge_detection/image", Image, queue_size=1)

    
    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg)
        # Resize the first frame for ROI selection
        resized_first_frame = cv2.resize(frame, (display_width, display_height))
        
        # Let the user select the ROI on the resized first frame
        roi = cv2.selectROI("Select ROI", resized_first_frame, False, False)
        cv2.destroyWindow("Select ROI")
        
        # Calculate the ROI on the original frame scale
        scale_x = frame.shape[1] / display_width
        scale_y = frame.shape[0] / display_height
        x, y = (int(roi[0] * scale_x), int(roi[1] * scale_y))
        w, h = (int(roi[2] * scale_x), int(roi[3] * scale_y))
        
        # Check if the ROI was actually selected
        if w and h:
            # Crop the frame to the scaled ROI
            roi_frame = frame[y:y+h, x:x+w]
    
            # # Convert ROI frame to grayscale
            gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            blurred_roi = cv2.GaussianBlur(gray_roi, gausian_blur_kernel_size, gausian_blur_kernel_sigma)
            edges = cv2.Canny(blurred_roi, canny_threshold_low, canny_threshold_high)
            kernel = np.ones((3,3), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=4)
            resized_edges_window = cv2.resize(dilated_edges, (edge_window_width, edge_window_height))
            fps_text = f"FPS: {fps:.2f}"
            Brightness_text = f"Brightness: {brightness:.2f}"
            position_fps = (40, 80)  # Adjust as needed
            position_brightness = (40, 160) # Adjust as needed
            cv2.putText(frame, fps_text, position_fps, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, Brightness_text, position_brightness, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            min_contours_length = 1500 # Set minimum contour length 
            long_contours =  [cnt for cnt in contours if cv2.arcLength(cnt, True) > min_contours_length]
            total_length = 0 
            total_length_meters = 0
            coordinates = []
            for i, cnt in enumerate(long_contours): 
                curvature = detect_curve(cnt)
                arc_length = cv2.arcLength(cnt, True)
                is_curve, label = detect_curve_geometry(cnt)
                cv2.putText(frame, label, (20, 2000), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 8)
                rails_coordinates.append(cnt)
                contour_coords = cnt.reshape(-1, 2)  # Reshape contour coordinates to a 2-column array
                coordinates.extend(contour_coords.flatten().tolist())  # Flatten and append coordinates
                msg = Float32MultiArray()
                msg.data = coordinates    
                print(coordinates)
                self.pub.publish(msg)
                print("Coordinates published successfully.")
    
    
                # Calculates perimeter or arc length of the contour 'cnt'. Second argument ('true') specifies the contour is closed 
                length = cv2.arcLength(cnt,True)
                total_length += length
                total_length_meters += length * pixel_to_meter_ratio
                cv2.drawContours(roi_frame, [cnt], -1, (0,0, 255), 15)
                # Display length on the frame
                cv2.putText(frame, f"Length {i+1}: {total_length_meters:.2f} meters", (10, 230 + i*50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            frame[y:y+h, x:x+w] = roi_frame
    
            # Resize the entire frame for display
            resized_frame = cv2.resize(frame, (display_width, display_height))
            if frame_count > 0: 
                average_brightness = total_brightness / frame_count
                print("Average Brightness of all frames: ", average_brightness)
            else: 
                print("No frames to process.")
                
# Function to detect curves
def detect_curve_geometry(cnt):
        # Approximate the contour to a simpler polygon
    epsilon = 0.01 * cv2.arcLength(cnt, True) # Smaller the number results to more points in the approximation, leading to closer fit of original contour. epsilon 0.01 means 1 %. 
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    # Compare the number of vertices in the approximation
    # A contour with a small number of vertices is likely to be a straight line
    # while a contour with a larger number of vertices is likely to be a curve
    if len(approx) > 2:  # Adjust this threshold based on your specific case
        return True, 'Curve'
    else:
        return False, 'Not curve'

# Quadratic curve function
def curve_func(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def detect_curve(coordinates):
    x = [point[0][0] for point in coordinates]
    y = [point[0][1] for point in coordinates]

    # Fit a quadratic curve to the coordinates
    popt, _ = curve_fit(curve_func, x, y)

    # Calculate curvature (second derivative of the curve function)
    curvature = 2 * popt[0]

    return curvature

def estimate_weather(brightness): 
# Estimates weather condition based on average brightness. 
# Returns a string indicating the condition ('Bright', 'Normal', 'Low')

    if brightness > 100: 
        return 'bright'
    elif brightness > 85: 
        return 'normal'
    else: 
        return 'low_light'



def main():
    rospy.init_node("edge_detection_node")
    node = EdgeDetectionNode("/camera/image")
    rospy.spin()

if __name__ == '__main__':
    exit(main())