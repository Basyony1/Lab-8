import cv2
import time
import numpy as np

def load_image(file_path, resize_dim):
    """
    Load an image from the specified file path and resize it.
    If the image cannot be loaded, raise an error.
    """
    img = cv2.imread(file_path)
    if img is None:
        raise FileNotFoundError(f"Image file not found at path: {file_path}")
    return cv2.resize(img, resize_dim)


def overlay_image(background, overlay, position):
    """
    Overlay an image with optional transparency (PNG) onto a background image.
    """
    x, y = position
    h, w = overlay.shape[:2]

    # Ensure the overlay fits within the background
    if x + w > background.shape[1] or y + h > background.shape[0]:
        return background

    # Check if the overlay has an alpha channel
    if overlay.shape[2] == 4:  # BGRA image
        b, g, r, alpha_channel = cv2.split(overlay)
        alpha = alpha_channel / 255.0  # Normalize alpha to [0, 1]
    else:  # BGR image (no alpha channel)
        b, g, r = cv2.split(overlay)
        alpha = np.ones_like(b)  # Create a fully opaque alpha channel

    roi = background[y:y+h, x:x+w]

    # Blend the overlay with the background
    for c in range(3):  # Iterate over RGB channels
        roi[:, :, c] = alpha * overlay[:, :, c] + (1 - alpha) * roi[:, :, c]

    background[y:y+h, x:x+w] = roi
    return background


def detect_and_track(frame, template_gray, fly_overlay, frame_width):
    """
    Detect the template in the frame and overlay the fly image.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.6
    loc = np.where(result >= threshold)

    has_match = False  # Flag to track if any matches are found

    for pt in zip(*loc[::-1]):
        x, y = pt
        w, h = template_gray.shape[1], template_gray.shape[0]

        # Only highlight matches in the right half of the frame
        if x + (w // 2) > frame_width // 2:
            has_match = True  # Set flag to True since a match is found
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            frame = overlay_image(frame, fly_overlay, (x, y))

            # Print coordinates every 5 frames
            print(f"Position: ({x + w//2}, {y + h//2})")

    return frame, has_match


def video_spot():
    """
    Main function to capture video, detect a reference point, and overlay a fly image.
    """
    try:
        # Initialize video capture and load images
        cap = cv2.VideoCapture(0)
        ref_point = load_image('ref-point.jpg', (100, 100))
        fly_overlay = load_image('fly64.png', (100, 100))
        ref_point_gray = cv2.cvtColor(ref_point, cv2.COLOR_BGR2GRAY)

        down_points = (640, 480)
        i = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera.")
                break

            frame = cv2.resize(frame, down_points, interpolation=cv2.INTER_LINEAR)
            frame, has_match = detect_and_track(frame, ref_point_gray, fly_overlay, frame.shape[1])

            # Display the frame
            cv2.imshow('frame', frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.1)
            i += 1

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    video_spot()