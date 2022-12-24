# ====== Sample Code for Smart Design Technology Blog ======

# Intel Realsense D435 cam has RGB camera with 1920×1080 resolution
# Depth camera is 1280x720
# FOV is limited to 69deg x 42deg (H x V) - the RGB camera FOV

# If you run this on a non-Intel CPU, explore other options for rs.align
    # On the NVIDIA Jetson AGX we build the pyrealsense lib with CUDA

import pyrealsense2 as rs
import mediapipe as mp
import cv2
import numpy as np
import datetime as dt
import math

font = cv2.FONT_HERSHEY_SIMPLEX
org = (20, 100)
fontScale = .5
color = (0,50,255)
thickness = 1

#Matrices
projection_matrix = np.array([
    [1, 0, 0],
    [0 ,1, 0],
    [0, 0, 1]])


#Parameters in meters
class Prop:
    def __init__(self, x, y, z, width, length, height):
        #Initialize Base Coordinates
        self.x = x
        self.y = y
        self.z = z
        
        #Initialize Base Dimensions
        self.width = width
        self.length = length
        self.height = height
        
        #initialling cube points
        self.obj = []
        self.obj.append(np.array([-1*width/2, -1*length/2, -1*height/2]))#B0
        self.obj.append(np.array([width/2, -1*length/2, -1*height/2]))#Bx
        self.obj.append(np.array([-1*width/2, length/2, -1*height/2]))#By
        self.obj.append(np.array([width/2, length/2, -1*height/2]))#Bxy

        self.obj.append(np.array([-1* width/2, -1*length/2, height/2]))#T0
        self.obj.append(np.array([width/2, -1*length/2, height/2]))#Tx
        self.obj.append(np.array([-1* width/2, length/2, height/2]))#Ty
        self.obj.append(np.array([width/2, length/2, height/2]))#Txy
        
        #Making line/point pairs for the cube
        self.lines = []
        self.lines.append([0,1]) #B0-Bx
        self.lines.append([0,2]) #B0-By
        self.lines.append([3,1]) #Bxy-Bx
        self.lines.append([3,2]) #Bxy-By

        self.lines.append([4,5]) #T0-Tx
        self.lines.append([4,6]) #T0-Ty
        self.lines.append([7,5]) #Txy-Tx
        self.lines.append([7,6]) #Txy-Ty

        self.lines.append([0,4]) #B0-T0
        self.lines.append([1,5]) #Bx-Tx
        self.lines.append([2,6]) #By-Ty
        self.lines.append([3,7]) #Bxy-Txy
        
        self.color = (0,255,0)
        
        #Collision (Sphere)
        self.radius = max(length, width, height)

#Rotation functions
    def rotate_z (self, angle):
        for index, point in enumerate(self.obj):
            rotation_z = np.array([
                [math.cos(angle), -1*math.sin(angle), 0],
                [math.sin(angle), math.cos(angle), 0],
                [0, 0, 1],
            ])
            rotation_z = np.dot(rotation_z, point.reshape((3,1)))
            self.obj[index] = rotation_z

    def rotate_y (self, angle):
        for index, point in enumerate(self.obj):
            rotation_y = np.array([
                [math.cos(angle), 0, math.sin(angle)],
                [0, 1, 0],
                [-1*math.sin(angle), 0, math.cos(angle)],
            ])
            rotation_y = np.dot(rotation_y, point.reshape((3,1)))
            self.obj[index] = rotation_y
            
    def rotate_x (self, angle):
        for index, point in enumerate(self.obj):
            rotation_x = np.array([
                [1, 0, 0],
                [0, math.cos(angle), -1*math.sin(angle)],
                [0, math.sin(angle), math.cos(angle)],
            ])
            rotation_x = np.dot(rotation_x, point.reshape((3,1)))
            self.obj[index] = rotation_x
    
    def translate_x (self, dist):
        self.x = self.x + dist
    
    def translate_y (self, dist):
        self.y = self.y + dist
    
    def translate_z (self, dist):
        temp = self.z
        self.z = self.z + dist
        if self.z < 0.05: #z coordinate must be greater than 0.05 (prevent division by 0 error)
            self.z = temp
        

    

    def project_2d (self, depth_image_flipped, depth_scale):
        projected_points = []
        for index, point in enumerate(self.obj):
            obj_x = point[0] + self.x
            obj_y = point[1] + self.y
            obj_z = point[2] + self.z
            
            #X projection 
            mid_x = 640
            fov_x = 87.0 / 180.0 * 3.14159
            full_x = obj_z * math.tan(fov_x/2) #Full width given distance obj_z
            proj_x = 640 + int(obj_x / full_x * 640)
            
            
            #Y projection
            mid_y = 360
            fov_y = 58.0 / 180.0 * 3.14159
            full_y = obj_z * math.tan(fov_y/2) #Full width given distance obj_z
            proj_y = 360 - int(obj_y / full_y * 360)
                        
            projected_points.append([proj_x,proj_y])
                
        
        return projected_points
    
    def projected_center (self):
        obj_x = self.x
        obj_y = self.y
        obj_z = self.z
        
        #X projection 
        mid_x = 640
        fov_x = 87.0 / 180.0 * 3.14159
        full_x = obj_z * math.tan(fov_x/2) #Full width given distance obj_z
        proj_x = 640 + int(obj_x / full_x * 640)
        
        
        #Y projection
        mid_y = 360
        fov_y = 58.0 / 180.0 * 3.14159
        full_y = obj_z * math.tan(fov_y/2) #Full width given distance obj_z
        proj_y = 360 - int(obj_y / full_y * 360)
                    
        return [proj_x,proj_y]
        


#def project_cube (object):
#    projections = object.project_2d()
#    for connect in lines:
#        start = projections[connect[0]]
#        end = projections[connect[1]]
#        
#        start_x =  start[0]
#        start_y = start[1]
#        
#        end_x = end[0]
#        end_y = end[1]
#        
#        images = cv2.line(images, (start_x, start_y), (end_x, end_y), (0, 255, 0), thickness)


#Making a cube test
cube = Prop(0, 0, 0.5, 0.2, 0.2, 0.2)
print(cube.radius)






# ====== Realsense ======
realsense_ctx = rs.context()
connected_devices = [] # List of serial numbers for present cameras
for i in range(len(realsense_ctx.devices)):
    detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
    print(f"{detected_camera}")
    connected_devices.append(detected_camera)
device = connected_devices[0] # In this example we are only using one camera
pipeline = rs.pipeline()
config = rs.config()
background_removed_color = 153 # Grey

# ====== Mediapipe ======
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


# ====== Enable Streams ======
config.enable_device(device)

# # For worse FPS, but better resolution:
# stream_res_x = 1280
# stream_res_y = 720
# # For better FPS. but worse resolution:
stream_res_x = 1280   
stream_res_y = 720

stream_fps = 30

config.enable_stream(rs.stream.depth, stream_res_x, stream_res_y, rs.format.z16, stream_fps)
config.enable_stream(rs.stream.color, stream_res_x, stream_res_y, rs.format.bgr8, stream_fps)
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

# ====== Get depth Scale ======
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"\tDepth Scale for Camera SN {device} is: {depth_scale}")

# ====== Set clipping distance ======
clipping_distance_in_meters = 2
clipping_distance = clipping_distance_in_meters / depth_scale
print(f"\tConfiguration Successful for SN {device}")

# ====== Get and process images ====== 
print(f"Starting to capture images on SN: {device}")

while True:
        
    
    
    
    start_time = dt.datetime.today().timestamp()

    # Get and align frames
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    
    if not aligned_depth_frame or not color_frame:
        continue

    # Process images
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    depth_image_flipped = cv2.flip(depth_image,1)
    color_image = np.asanyarray(color_frame.get_data())

    depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #Depth image is 1 channel, while color image is 3
    background_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), background_removed_color, color_image)

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    images = cv2.flip(background_removed,1)
    color_image = cv2.flip(color_image,1)
    color_images_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    
    #Project 3D Obj
    touchedCube = False
    #cube.translate_z(0.01)
    projections = cube.project_2d(depth_image_flipped, depth_scale)
    z_values = []
    for point in cube.obj:
        z_values.append(point[2])
    
    for connect in cube.lines:
        start = projections[connect[0]]
        end = projections[connect[1]]
        
        start_x = start[0]
        start_y = start[1] 
        
        end_x = end[0]
        end_y = end[1]
        
        if (start_x >= 0 and start_x <= 1280) and (start_y >= 0 and start_y <= 720) and (end_x >= 0 and end_x <= 1280) and (end_y >= 0 and end_y <= 720):
            cover_depth_start = depth_image_flipped[start_y,start_x] * depth_scale # meters
            cover_depth_end = depth_image_flipped[end_y,end_x] * depth_scale # meters
            
            if z_values[connect[0]] > cover_depth_start or z_values[connect[1]] > cover_depth_end:
                images = cv2.line(images, (start_x, start_y), (end_x, end_y), (0, 50, 0), 1)
            else:
                images = cv2.line(images, (start_x, start_y), (end_x, end_y), cube.color, 3)
        else:
            images = cv2.line(images, (start_x, start_y), (end_x, end_y), cube.color, 1)
        
        
    
    
    
    hand_parts = ["wrist,", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip", "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip", "middle_finger_mpc", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip", "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip", "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"]
    # Process hands
    results = hands.process(color_images_rgb)
    if results.multi_hand_landmarks:
        number_of_hands = len(results.multi_hand_landmarks)
        i=0
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(images, handLms, mpHands.HAND_CONNECTIONS)
            org2 = (20, org[1]+(20*(i+1)))
            hand_side_classification_list = results.multi_handedness[i]
            hand_side = hand_side_classification_list.classification[0].label
            
            #Finds General hand distance and adds text + landmark circles to the hands
            middle_finger_knuckle = results.multi_hand_landmarks[i].landmark[9]
            x = int(middle_finger_knuckle.x*len(depth_image_flipped[0]))
            y = int(middle_finger_knuckle.y*len(depth_image_flipped))
            if x >= len(depth_image_flipped[0]):
                x = len(depth_image_flipped[0]) - 1
            if y >= len(depth_image_flipped):
                y = len(depth_image_flipped) - 1
            mfk_distance = depth_image_flipped[y,x] * depth_scale # meters
            mfk_distance_feet = mfk_distance * 3.281 # feet
            images = cv2.putText(images, f"{hand_side} Hand Distance: {mfk_distance_feet:0.3} feet ({mfk_distance:0.3} m) away", org2, font, fontScale, color, thickness, cv2.LINE_AA)
            
            #Calcualtions for x in meters
            z = depth_image_flipped[y,x] * depth_scale
            mid_x = 1280 / 2
            horz_dist = x - mid_x
            fov_x = 87.0 / 180.0 * 3.14159
            real_dist_x =  z * (math.tan(fov_x/2))*(horz_dist / mid_x)
            #print(str(real_dist_x) + "m for x")
            
            #Calcualtions for y in meters
            mid_y = 720 / 2
            vert_dist = mid_y - y 
            fov_y = 58.0 / 180.0 * 3.14159
            real_dist_y =  z * (math.tan(fov_y/2))*(vert_dist / mid_y)
            #print(str(real_dist_y) + "m for y")
            

            
            #Projection line testing

            cube.rotate_x(0.01)
            cube.rotate_z(0.01)
            cube.rotate_y(0.01)
            
            #Calculates the x, y, and depth values for each hand_part
            #print("hand: " + str(i))
            for j in range(21):
                cur_hand = results.multi_hand_landmarks[i].landmark[j]
                x = int(cur_hand.x*len(depth_image_flipped[0]))
                y = int(cur_hand.y*len(depth_image_flipped))
                if x >= len(depth_image_flipped[0]):
                    x = len(depth_image_flipped[0]) - 1
                if y >= len(depth_image_flipped):
                    y = len(depth_image_flipped) - 1
                z = depth_image_flipped[y,x] * depth_scale # meters
                #print(hand_parts[j] + ": " + str(x) + " " + str(y) + " " + str(z))

                mid_x = 1280 / 2
                horz_dist = x - mid_x
                fov_x = 87.0 / 180.0 * 3.14159
                real_dist_x =  z * (math.tan(fov_x/2))*(horz_dist / mid_x)
                #print(str(real_dist_x) + "m for x")
                
                #Calcualtions for y in meters
                mid_y = 720 / 2
                vert_dist = mid_y - y 
                fov_y = 58.0 / 180.0 * 3.14159
                real_dist_y =  z * (math.tan(fov_y/2))*(vert_dist / mid_y)
                #print(str(real_dist_y) + "m for y")
                
                
                
                distance = math.sqrt((cube.x - real_dist_x)**2 + (cube.y - real_dist_y)**2 + (cube.z - z)**2) 
                print(distance)
                #TEST
                if distance <= cube.radius * 0.90:
                    touchedCube = True
                    for k in range(100):
                        cube.translate_x(1/distance * (cube.x - real_dist_x) * 0.0001)
                        cube.translate_y(1/distance * (cube.y - real_dist_y) * 0.0001)
                        cube.translate_z(1/distance * (cube.z - z) * 0.0001)
                    
            
            #print("\n")
            
            proj_center = cube.projected_center()
            images = cv2.putText(images, (str(round(cube.z, 3)) + "m"), (proj_center[0] + 50 ,proj_center[1] + 50), font, fontScale, color, thickness, cv2.LINE_AA)
                    
        
            
            i+=1
        
        if touchedCube == True:


            cube.color = (255,0,0)
                
            
        else:
            images = cv2.putText(images, f"Hands: {number_of_hands}", org, font, fontScale, color, thickness, cv2.LINE_AA)
            cube.color = (0,255,0)
        
    else:
        images = cv2.putText(images,"No Hands", org, font, fontScale, color, thickness, cv2.LINE_AA)
        cube.color = (0,255,0)

    # Display FPS
    time_diff = dt.datetime.today().timestamp() - start_time
    fps = int(1 / time_diff)
    org3 = (20, org[1] + 60)
    images = cv2.putText(images, f"FPS: {fps}", org3, font, fontScale, color, thickness, cv2.LINE_AA)
    

   
    

    name_of_window = 'SN: ' + str(device)

    # Display images 
    cv2.namedWindow(name_of_window, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name_of_window, images)
    key = cv2.waitKey(1)
    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        print(f"User pressed break key for SN: {device}")
        break

print(f"Application Closing")
pipeline.stop()
print(f"Application Closed.")