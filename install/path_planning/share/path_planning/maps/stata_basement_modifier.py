# cd home/racecar_ws/src/path_planning/install/path_planning/share/path_planning/maps

from PIL import Image
import numpy as np
import cv2

# Load the map
map_image = Image.open('stata_basement.png').convert('L')  # Convert to grayscale
map_array = np.array(map_image)

# Step 1: Set non-white to black and vice versa
map_array = np.where(map_array < 255, 0, 255).astype(np.uint8)

# Step 2: Invert colors (black to white, white to black)
map_array = np.where(map_array == 255, 0, 255).astype(np.uint8)

# Step 3: Dilate the white areas
kernel = np.ones((12,12),np.uint8)
map_array = cv2.dilate(map_array, kernel, iterations = 1)

# Step 4: Flip colors back
map_array = np.where(map_array == 255, 0, 255).astype(np.uint8)

# Save the processed map
new_map_image = Image.fromarray(map_array)
new_map_image.save('processed_map.png')
new_map_image.save('path_on_map.png')


import json

def world_to_pixel(world_xy, origin = (25.9, 48.5), resolution = 0.0504):
    world_x, world_y = world_xy
    # Convert world coordinates (meters) to pixel coordinates
    pixel_x = -int((world_x - origin[0]) / resolution)
    pixel_y = int((world_y - origin[1]) / resolution) + 1300
    return pixel_x, pixel_y


def load_trajectory(file_path):
    # Open the file and load the data
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Extract the points
    points = data['points']
    trajectory = [(point['x'], point['y']) for point in points]
    return trajectory

for i in range(7):
    # Usage
    trajectory = load_trajectory(f'../lanes/lane-segment-{str(i+1)}.traj')
    # Example usage

    # Drawing the trajectory onto the map
    map_image = cv2.imread('path_on_map.png')
    for j in range(len(trajectory)-1):
        start_point = world_to_pixel(trajectory[j])
        end_point = world_to_pixel(trajectory[j+1])
        print(start_point, end_point)
        if i < 3:
            cv2.line(map_image, start_point, end_point, color=(0,0,0), thickness=12)
        else:
            cv2.line(map_image, start_point, end_point, color=(0,0,0), thickness=5)
        # map_array[int(y), int(x)] = 0  # Set trajectory pixels to black (obstacles)

    cv2.imwrite('path_on_map.png', map_image)
    # Save the map with the trajectory
    # new_map_image = Image.fromarray(map_array)
    # new_map_image.save('map_with_trajectory.png')