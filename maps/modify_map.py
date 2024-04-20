import cv2
import numpy as np
from os.path import exists
import networkx as nx


def process_image(image_path, thickness=2):
    """
    This function takes a grayscale image path and a thickness value. 
    It changes every pixel that is not completely black into white, 
    makes the black lines thicker by the given number of pixels, 
    and then saves the new image.
    """
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image has been properly loaded
    if image is None:
        return "The image could not be loaded. Please check the file path."
    
    # Any pixel that is not black (0) is set to white (255)
    x = 200 # 200 for gray is like white, 250 for gray is like black
    binary_image = np.where(image > 254, 0, 255).astype(np.uint8)
    
    # Define the structural element for the morphological operations
    kernel = np.ones((thickness, thickness), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
    thicker_lines_image = cv2.dilate(binary_image, kernel, iterations=1)
    end_result = np.where(thicker_lines_image > 0, 0, 255)
    
    # Make the black lines thicker
    #thicker_lines_image = cv2.dilate(binary_image, kernel, iterations=1)
    
    # Save the processed image
    processed_image_path = image_path.replace('.png', '_thickened.png')
    cv2.imwrite(processed_image_path, end_result)
    
    return end_result, processed_image_path



def create_graph_from_white_pixels(image, include_diagonals=False):
    """
    Create a graph from an image where each white pixel is connected to its
    neighboring white pixels. Optionally include diagonal connections.
    """
    # Ensure the image is in binary format (i.e., only contains 0s and 255s)
    # image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]

    # Initialize the graph
    G = nx.Graph()
    
    # Get the dimensions of the image
    rows, cols = image.shape
    
    # Define the directions for neighbors (8 directions if diagonals included, else 4)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    if include_diagonals:
        directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])  # Diagonals
    
    # Iterate over the image pixels
    for x in range(rows):
        for y in range(cols):
            # Check if the current pixel is white
            if image[x, y] == 255:
                # Add the current pixel as a node to the graph
                G.add_node((x, y))
                # Check for white neighbors
                for dx, dy in directions:
                    neighbor_x, neighbor_y = x + dx, y + dy
                    # Check if the neighbor coordinates are inside the image bounds
                    if 0 <= neighbor_x < rows and 0 <= neighbor_y < cols:
                        # Check if the neighbor is white
                        if image[neighbor_x, neighbor_y] == 255:
                            # Add the neighbor as a node and connect it to the current pixel
                            G.add_node((neighbor_x, neighbor_y))
                            G.add_edge((x, y), (neighbor_x, neighbor_y))
    return G

# # Load the image (it's assumed that the image is already in grayscale format)
# image_path = '/mnt/data/Screenshot 2024-04-20 at 12.27.32 AM_processed.png'
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # Create a graph from the white pixels of the image
# # Set 'include_diagonals' to True if diagonal connections are desired
# graph = create_graph_from_white_pixels(image, include_diagonals=True)

# # Let's get some basic information about the graph
# num_nodes = graph.number_of_nodes()
# num_edges = graph.number_of_edges()

# num_nodes, num_edges



if __name__ == '__main__':
    print("processing stata basement")
    print(exists('maps/stata_basement.png'))
    processed_image, processed_image_path = process_image('maps/stata_basement.png', thickness=18)
    graph = create_graph_from_white_pixels(processed_image, include_diagonals=True)
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    print(num_nodes, num_edges)
    