---
name: opencv-image-reader
description: Use this skill to read, load, or open an image using OpenCV in Python when you need to process images in RGB, BGR, or grayscale format. Trigger this skill with phrases like "load image with OpenCV", "read image using cv2", "convert image to RGB", "display image with OpenCV", "inspect image shape", or "load image in grayscale".
---

# OpenCV Image Reader

## Overview
The OpenCV Image Reader skill allows users to read images in various formats (RGB, BGR, grayscale) using OpenCV in Python. This skill can be used to load an image from a file path, convert between color spaces, display images, and inspect image properties.

## Automatic Processing
This skill automatically handles the following tasks:
- Reading images from file paths
- Converting images between BGR, RGB, and grayscale formats
- Displaying images using OpenCV's imshow function
- Inspecting image properties such as shape and data type

## Core Capabilities
- Read images from file paths using OpenCV's imread function
- Convert images between BGR, RGB, and grayscale color spaces
- Display images using OpenCV's imshow function
- Inspect image properties such as shape and data type

## Workflow
1. **Read Image**: Use OpenCV's `imread` function to read an image from a provided file path. The function takes two arguments: the file path and a flag indicating the desired color space (BGR, RGB, or grayscale).
2. **Convert Color Space (Optional)**: If the user requests a different color space, use OpenCV's `cvtColor` function to convert the image.
3. **Display Image (Optional)**: If the user wants to display the image, use OpenCV's `imshow` function to display the image in a window.
4. **Inspect Image Properties (Optional)**: If the user wants to inspect image properties, use Python's built-in functions such as `shape` and `dtype` to retrieve the image's shape and data type.
5. **Return Image Data**: Return the image data in the requested color space.

## Usage Patterns
### Pattern 1: Basic
- Input: `image_file_path = "path/to/image.jpg"`
- Output: Image data in BGR format
- Usage: `read_image(image_file_path)`

### Pattern 2: Advanced
- Input: `image_file_path = "path/to/image.jpg"`, `color_space = "RGB"`
- Output: Image data in RGB format
- Usage: `read_image(image_file_path, color_space="RGB")`

## Error Handling
| Error | Cause | User Message |
| --- | --- | --- |
| FileNotFoundError | Invalid file path | "The file path is invalid. Please provide a valid file path." |
| ValueError | Invalid color space | "Invalid color space. Please choose from BGR, RGB, or grayscale." |

## Output Formatting
- Short content: Image data in BGR, RGB, or grayscale format
- Medium content: Image data with shape and data type information
- Long content: Detailed image properties, including shape, data type, and color space

## Best Practices
- Always validate the file path and color space inputs
- Use try-except blocks to handle potential errors
- Use OpenCV's built-in functions for image processing and display
- Use Python's built-in functions for inspecting image properties