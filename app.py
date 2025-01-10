import os
import numpy as np
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from io import BytesIO

# Load the pre-trained model
MODEL_PATH = 'traffic_sign_classifier.h5'
model = load_model(MODEL_PATH)


# Function to download an image from the provided URL
def download_image(url):
    try:
        response = requests.get(url, timeout=10)  # Download the image with a 10-second timeout
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return img
        else:
            print(f"Error: Unable to fetch the image. HTTP Status Code: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred while downloading the image: {e}")
        return None


# Function to classify the downloaded image
def classify_image_from_url(img_url):
    try:
        # Download the image
        img = download_image(img_url)
        if img is None:
            return None

        # Resize and preprocess the image
        img = img.resize((32, 32))  # Resize to the expected input shape
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Perform a prediction
        prediction = model.predict(img_array)
        class_id = np.argmax(prediction)
        confidence = prediction[0][class_id] * 100
        return class_id, confidence
    except Exception as e:
        print(f"An error occurred during image classification: {e}")
        return None


# Main function for the console application
def main():
    print("Welcome to the Traffic Sign Classifier Console Application!")
    print("Please make sure your model ('traffic_sign_classifier.h5') is in the same directory.")
    while True:
        print("\nMenu:")
        print("1. Classify an Image from URL")
        print("2. Exit")
        choice = input("Enter your choice (1/2): ").strip()

        if choice == '1':
            img_url = input("Enter the URL of the image to classify: ").strip()
            result = classify_image_from_url(img_url)
            if result:
                class_id, confidence = result
                print(f"\nClassification Result:")
                print(f"Class ID: {class_id}")
                print(f"Confidence: {confidence:.2f}%")
            else:
                print("Failed to classify the image. Please check the URL and try again.")
        elif choice == '2':
            print("Exiting the application. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")


if __name__ == '__main__':
    main()
