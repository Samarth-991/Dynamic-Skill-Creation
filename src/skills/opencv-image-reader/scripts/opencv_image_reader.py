import cv2
import numpy as np
from typing import Dict

def read_image(image_path: str) -> Dict:
    """
    Reads an image from a file path using OpenCV.

    Args:
    image_path (str): The path to the image file.

    Returns:
    Dict: A dictionary containing the image data in BGR, RGB, and grayscale format.
    """
    try:
        # Read the image in BGR format (OpenCV default)
        bgr_image = cv2.imread(image_path)

        if bgr_image is None:
            return {"success": False, "error": "Failed to read image"}

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # Convert BGR to grayscale
        grayscale_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        return {
            "success": True,
            "bgr": bgr_image.tolist(),
            "rgb": rgb_image.tolist(),
            "grayscale": grayscale_image.tolist(),
            "image_properties": {
                "height": bgr_image.shape[0],
                "width": bgr_image.shape[1],
                "channels": bgr_image.shape[2] if len(bgr_image.shape) == 3 else 1
            }
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def display_image(image: np.ndarray) -> None:
    """
    Displays an image using OpenCV.

    Args:
    image (np.ndarray): The image to display.
    """
    try:
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error displaying image: {str(e)}")


def run_opencv_image_reader(input_value: str) -> Dict:
    """
    Runs the OpenCV image reader.

    Args:
    input_value (str): The path to the image file.

    Returns:
    Dict: A dictionary containing the result of the image read operation.
    """
    return read_image(input_value)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenCV Image Reader")
    parser.add_argument("image_path", help="The path to the image file")
    args = parser.parse_args()

    result = run_opencv_image_reader(args.image_path)

    if result["success"]:
        print("Image read successfully:")
        print(f"BGR: {result['bgr'][:5]}...")
        print(f"RGB: {result['rgb'][:5]}...")
        print(f"Grayscale: {result['grayscale'][:5]}...")
        print(f"Image Properties: {result['image_properties']}")

        # Display the BGR image
        bgr_image = np.array(result["bgr"])
        display_image(bgr_image)
    else:
        print(f"Error: {result['error']}")