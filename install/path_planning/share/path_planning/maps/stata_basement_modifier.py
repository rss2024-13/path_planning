# cd home/racecar_ws/src/path_planning/install/path_planning/share/path_planning/maps

from PIL import Image
import numpy as np
import cv2

if False:
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
            if i == 0:
                cv2.line(map_image, start_point, end_point, color=(0,0,0), thickness=8)
            elif i == 1:
                cv2.line(map_image, start_point, end_point, color=(0,0,0), thickness=10)
            elif i == 2:
                cv2.line(map_image, start_point, end_point, color=(0,0,0), thickness=12)
            else:
                cv2.line(map_image, start_point, end_point, color=(0,0,0), thickness=5)
            # map_array[int(y), int(x)] = 0  # Set trajectory pixels to black (obstacles)

        cv2.imwrite('path_on_map.png', map_image)
        # Save the map with the trajectory
        # new_map_image = Image.fromarray(map_array)
        # new_map_image.save('map_with_trajectory.png')


def convert_image_to_occupancy(image_path, output_path):
    # Load the RGB image
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image")
        return

    # Convert to grayscale for easy white and black detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Rotate the image 180 degrees
    gray_image = cv2.rotate(image, cv2.ROTATE_180)

    # Initialize the occupancy grid
    occupancy_grid = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255

    # Extract RGB channels
    red_channel = image[:, :, 2]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 0]

    # Define masks for white and black
    white_mask = (gray_image == 255)
    black_mask = (gray_image == 0)

    # Assign values for white and black
    occupancy_grid[white_mask] = 100    # White
    occupancy_grid[black_mask] = 0    # Black

    # Determine color dominance for non-white and non-black pixels
    non_extreme_colors = ~(white_mask | black_mask)
    red_dominant = non_extreme_colors & (red_channel > green_channel) & (red_channel > blue_channel)
    green_dominant = non_extreme_colors & (green_channel > red_channel) & (green_channel > blue_channel)
    blue_dominant = non_extreme_colors & (blue_channel > red_channel) & (blue_channel > green_channel)

    # Assign grayscale values according to the dominance
    occupancy_grid[red_dominant] = int(100/3 * 1)
    occupancy_grid[green_dominant] = int(100/3 * 2)
    occupancy_grid[blue_dominant] = 0

    print(occupancy_grid.flatten())
    np.savetxt('occupancy_grid.txt', occupancy_grid)
    

    # Save the modified image
    cv2.imwrite(output_path, occupancy_grid)

    print(f"Occupancy grid saved to {output_path}")


convert_image_to_occupancy('path_on_map_2.png', 'updated_path_on_map.png')