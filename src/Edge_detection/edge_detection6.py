#!/usr/bin/env /usr/bin/python3

import cv2
import numpy as np
from scipy.optimize import curve_fit

# ROS dependencies
import rospy
from std_msgs.msg import Float32MultiArray

video_name = '/home/cwlcp99/catkin_ws/src/edge_detection/src/Python_algo/Test_10_25mph_sunset.mp4'

# Capture video from file or camera
cap = cv2.VideoCapture(video_name, cv2.CAP_ANY)  # CAP_ANY to let OpenCV automatically choose the backend

# Get fps from the original video
fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS of the video: ", fps)

# Desired frame size for display
display_width = 1280
display_height = 720

# Define the size for the edge detection window 
edge_window_width = 540
edge_window_height = 360

# Initialize variables for calculating average brightness
total_brightness = 0
frame_count = 0 

# Initialize a list to store coordinates before starting the frame processing loop
rails_coordinates = []

# Threshold to detect curves 
curve_threshold = 0.01

# Define the pixel-to-meter ratio
pixel_to_meter_ratio = 0.02  # Example value, adjust based on your calibration

# Initialize ROS node and publisher
rospy.init_node('railway_coordinates_publisher', anonymous=True)
coordinates_publisher = rospy.Publisher('railway_coordinates', Float32MultiArray, queue_size=10)

fourcc = cv2.VideoWriter_fourcc(*'mp4v') #Codec for mp4 format 
out = cv2.VideoWriter('Results\outputvideo_test10.mp4', fourcc, fps, (display_width, display_height))

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

# Get the first frame of the video
ret, frame = cap.read()
if not ret:
    cap.release()
    raise RuntimeError("Failed to read video")

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
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame to the scaled ROI
        roi_frame = frame[y:y+h, x:x+w]

        # Convert ROI frame to grayscale
        gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        
         # Calculate the brightness of the frame
        brightness = np.mean(gray_roi)
        total_brightness += brightness
        frame_count += 1
        
        # Dynamic parameter adjuster depending on brightness.
        weather_condition = estimate_weather(brightness)
        
        if weather_condition == 'bright': # Bright lightning conditions
            gausian_blur_kernel_size = (3,3)
            gausian_blur_kernel_sigma = 1
            canny_threshold_low = 50 
            canny_threshold_high = 150 
        elif weather_condition == 'normal': # Normal lightning conditions
            gausian_blur_kernel_size = (5,5)
            gausian_blur_kernel_sigma = 1
            canny_threshold_low = 70 
            canny_threshold_high = 150
        else: # Low light conditions
            gausian_blur_kernel_size = (5,5)
            gausian_blur_kernel_sigma = 2
            canny_threshold_low = 50 
            canny_threshold_high = 100
        
         # Apply Gaussian Blur to smooth the image
         # 0 for test 10
         # 2 for test 2, 3, 4, 5, 6, 7, 8, 9, 10 
        blurred_roi = cv2.GaussianBlur(gray_roi, gausian_blur_kernel_size, gausian_blur_kernel_sigma)

        # Apply edge detection to the ROI
        # 50 and 150 for test 10 
        # 50 and 100 for test 2, 3, 4, 5, 6, 7, 8, 9, 10 
        edges = cv2.Canny(blurred_roi, canny_threshold_low, canny_threshold_high)
        
        # Dilate the edges to make them more visible
        kernel = np.ones((3,3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=4)
        
        resized_edges_window = cv2.resize(dilated_edges, (edge_window_width, edge_window_height))
        
        # Prepare the text to display (FPS)
        fps_text = f"FPS: {fps:.2f}"
        Brightness_text = f"Brightness: {brightness:.2f}"
        
        # Position the text on the frame (top-left corner)
        position_fps = (40, 80)  # Adjust as needed
        position_brightness = (40, 160) # Adjust as needed
        
        # Add text to the frame
        cv2.putText(frame, fps_text, position_fps, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(frame, Brightness_text, position_brightness, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # Find contours of the edges 
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on their length 
        # 900 min_contours_length for test 10
        min_contours_length = 1500 # Set minimum contour length 
        long_contours =  [cnt for cnt in contours if cv2.arcLength(cnt, True) > min_contours_length]
    
        
        # Initialize a variable to store the total lenght of contours 
        total_length = 0 
        
        # Initialize variable for total length of contours in meters
        total_length_meters = 0

        # After the loop where contours are processed
        coordinates = []

        
        # Enumerate() to get both the index (i) and the contour (cnt) for each iteration. 
        for i, cnt in enumerate(long_contours): 
            # Detect curve for the current contour
            curvature = detect_curve(cnt)
            
            # Calculate the arc length of the contour
            arc_length = cv2.arcLength(cnt, True)
          
            # Detect curve for the current contour
            is_curve, label = detect_curve_geometry(cnt)
            
            #if is_curve:
            #    label = 'Curve'
            #else:
            #    label = 'Not curve'
            
            # print(f'Contour {i}: {label}')
            
             # Add the classification label as text on the contour image
            cv2.putText(frame, label, (20, 2000), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 8)

            # Append the coordinates of each contour to the list
            rails_coordinates.append(cnt)

            #if rails_coordinates:
            # Convert coordinates to a Float32MultiArray message
            #    msg = Float32MultiArray()
            #    msg.data = np.array(rails_coordinates).flatten().tolist()

            # Extract x and y coordinates of the contour
            contour_coords = cnt.reshape(-1, 2)  # Reshape contour coordinates to a 2-column array
            coordinates.extend(contour_coords.flatten().tolist())  # Flatten and append coordinates

            # Convert coordinates to a Float32MultiArray message
            msg = Float32MultiArray()
            msg.data = coordinates

            print(coordinates)


            # Publish the coordinates
            coordinates_publisher.publish(msg)

            # Print a message after publishing
            print("Coordinates published successfully.")


            # Calculates perimeter or arc length of the contour 'cnt'. Second argument ('true') specifies the contour is closed 
            length = cv2.arcLength(cnt,True)
            total_length += length
            total_length_meters += length * pixel_to_meter_ratio
            cv2.drawContours(roi_frame, [cnt], -1, (0,0, 255), 15)
            # Display length on the frame
            cv2.putText(frame, f"Length {i+1}: {total_length_meters:.2f} meters", (10, 230 + i*50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # Place the processed ROI back into the original frame
        frame[y:y+h, x:x+w] = roi_frame

        # Resize the entire frame for display
        resized_frame = cv2.resize(frame, (display_width, display_height))
        
        # Write the frame with red edges to the output video file 
        out.write(resized_frame)

        # Display the resized frame with the red edges overlay
        cv2.imshow('Railway Edge Detection', resized_frame)
        
        # Display Edge detection frames 
        cv2.imshow('Edge detection', resized_edges_window)
        
        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

    # Calculate the average brightness of all frames 
    if frame_count > 0: 
        average_brightness = total_brightness / frame_count
        print("Average Brightness of all frames: ", average_brightness)
    else: 
        print("No frames to process.")
        
    # Print the coordinates after processing all contours in the frame
    #print("Coordinates after processing frame:", rails_coordinates)
        
    # Print the coordinates after processing all contours in the frame
    #print("Coordinates after processing frame:", rails_coordinates)


    # Release the video capture object and close all OpenCV windows
    out.release()
    cap.release()
    cv2.destroyAllWindows()
