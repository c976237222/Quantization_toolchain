import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms

def convert_images_to_npy(input_directory, output_directory, limit=None):
    """
    Convert images in the input directory to .npy format, preprocess them using PyTorch transforms,
    and save them to the output directory.

    Args:
        input_directory (str): Path to the input directory containing images.
        output_directory (str): Path to the output directory for saving .npy files.
        limit (int, optional): Number of images to convert. If None, convert all images.
    """
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Get list of all image files in the input directory
    image_files = [f for f in os.listdir(input_directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Limit the number of files to process if specified
    if limit is not None:
        image_files = image_files[:limit]

    # Define the transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Process each image file
    for image_file in tqdm(image_files, desc="Converting images", unit="file"):
        # Construct full input and output paths
        input_path = os.path.join(input_directory, image_file)
        output_path = os.path.join(output_directory, os.path.splitext(image_file)[0] + ".npy")

        try:
            # Open image using PIL
            img = Image.open(input_path).convert("RGB")

            # Apply the transforms
            img_tensor = transform(img)

            # Convert tensor to numpy array
            img_npy = img_tensor.numpy()

            # Save the numpy array to the output directory
            np.save(output_path, img_npy)

            # Display the shape of the output image
            print(f"Processed {image_file}: shape {img_npy.shape}")

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

# Example usage
input_directory = "/home/wangsiyuan/neural-compressor/examples/onnxrt/image_recognition/vgg16/quantization/ptq_static/val_image"
output_directory = "/home/wangsiyuan/ppq/data1"
convert_images_to_npy(input_directory, output_directory, limit=640)
