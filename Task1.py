import cv2
from matplotlib import pyplot as plt

def load_image(file_path):
    """
    Load an image from the specified file path.
    If the image cannot be loaded, raise an error.
    """
    img = cv2.imread(file_path)
    if img is None:
        raise FileNotFoundError(f"Image file not found at path: {file_path}")
    return img


def split_channels(image):
    """
    Split the input image into its B, G, and R channels.
    """
    b, g, r = cv2.split(image)
    return b, g, r


def display_channel(channel, cmap='Blues', title="Channel"):
    """
    Display a single channel of an image using Matplotlib.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(channel, cmap=cmap)
    plt.title(title)
    plt.axis('off')  # Turn off axis labels
    plt.show()


def image_processing(file_path='picture.png'):
    """
    Main function to process and display the blue channel of an image.
    """
    try:
        # Step 1: Load the image
        img = load_image(file_path)
        
        # Step 2: Split the image into B, G, and R channels
        b, g, r = split_channels(img)
        
        # Step 3: Display the blue channel
        display_channel(b, cmap='Blues', title="Blue Channel")
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    # Call the image processing function
    image_processing()

    # Clean up OpenCV windows (if any were opened)
    cv2.waitKey(0)
    cv2.destroyAllWindows()