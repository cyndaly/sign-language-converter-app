import cv2
import numpy as np
import pyttsx3
import speech_recognition as sr
from keras.models import load_model
from moviepy.editor import VideoFileClip

# Initialize the speech engine
engine = pyttsx3.init()

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Load the pre-trained model
model = load_model('path_to_your_model.h5')

# Define the labels
labels = ['A', 'B', 'C', 'D', ...]  # Continue with the rest of the alphabet

def sign_to_text(frame):
    # Preprocess the frame (this needs to match how your model was trained)
    img = cv2.resize(frame, (64, 64))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Predict the label of the sign
    predictions = model.predict(img)
    label = labels[np.argmax(predictions)]
    return label

def video_to_text(video_path):
    text = ""
    clip = VideoFileClip(video_path)

    for frame in clip.iter_frames():
        sign = sign_to_text(frame)
        text += sign

    return text

def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

def speech_to_text():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print(f"Recognized: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return ""

def display_sign_language(text):
    # Placeholder for displaying the sign language
    # You can use videos/images corresponding to each character/word
    print(f"Displaying sign language for: {text}")

# Example usage:

# 1. Convert sign language video to text, then to speech
sign_video_path = 'path_to_sign_language_video.mp4'
text_from_video = video_to_text(sign_video_path)
print(f"Text from video: {text_from_video}")
text_to_speech(text_from_video)

# 2. Convert speech to text, then to sign language video
spoken_text = speech_to_text()
if spoken_text:
    display_sign_language(spoken_text)
