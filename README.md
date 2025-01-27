# Gözce: Audio Feedback and Detection System for the Visually Impaired
Gözce is a webcam-based system designed to assist visually impaired individuals by providing real-time object detection and audio feedback. This innovative solution uses deep learning technologies to improve accessibility and independence in daily life.

## Features
- ***Real-Time Object Detection***: Powered by YOLOv8 for fast and accurate results.
- ***Audio Notifications***: Uses Pyttsx3 to convert detected objects into speech.
- ***Webcam Integration***: Captures the environment in real-time and processes it on the go.
- ***Efficient Notification System***: Prevents repeating alerts for previously detected objects.
- ***Trained with COCO Dataset***: Supports detection of 80+ object classes.

## Technologies Used
- ***Programming Language***: Python
- ***Deep Learning Framework***: TensorFlow Lite
- ***Computer Vision***: OpenCV
- ***Text-to-Speech***: Pyttsx3

## How to Run TensorFlow Lite Models on Windows
### Step 1. Download and Install Anaconda
First, install Anaconda, which is a Python environment manager that greatly simplifies Python package management and deployment. Anaconda allows you to create Python virtual environments on your PC without interfering with existing installations of Python. Go to the Anaconda Downloads page and click the Download button.

When the download finishes, open the downloaded .exe file and step through the installation wizard. Use the default install options.

### Step 2. Set Up Virtual Environment and Directory
Go to the Start Menu, search for "Anaconda Command Prompt", and click it to open up a command terminal. We'll create a folder called tflite1 directly in the C: drive. (You can use any other folder location you like, just make sure to modify the commands below to use the correct file paths.) Create the folder and move into it by issuing the following commands in the terminal:
```python
mkdir C:\tflite1
cd C:\tflite1
```
Next, create a Python 3.9 virtual environment by issuing:
```python
conda create --name tflite1-env python=3.9
```
Enter "y" when it asks if you want to proceed. Activate the environment and install the required packages by issuing the commands below. We'll install TensorFlow, OpenCV, and a downgraded version of protobuf. TensorFlow is a pretty big download (about 450MB), so it will take a while.
```python
conda activate tflite1-env
pip install tensorflow opencv-python protobuf==3.20.*
```
### Step 3. Run TensorFlow Lite Model!
Alright! Now that everything is set up, running the TFLite model is easy. Just call one of the detection scripts and point it at your model folder with the --modeldir option. For example, to run your custom_model_lite model on a webcam, issue:
```python
python TFLite_detection_webcam.py --modeldir=custom_model_lite
```
### How It Works
- Launch the system and connect a webcam.
- The webcam captures real-time video.
- YOLOv8 detects objects in the video stream.
- Detected objects are announced through audio notifications.

## Contributors
***Müyesser Şenyüz***<br></br>
***Somaya Arab***
