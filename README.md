# A DEEP LEARNING APPROACH FOR RECOGNIZING AGE, EMOTION AND GENDER IN FACIAL EXPRESSIONS

## Project Description
This project implements an **Age, Emotion, and Gender Recognition System** using **Deep Learning**. It provides a user-friendly interface for uploading videos, detecting faces, and classifying emotions, gender, and age in real-time.

## Features
- Detects human faces in a video.
- Classifies emotions into **Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised**.
- Predicts **gender (Male/Female)**.
- Estimates the **age group** of the detected face.
- Provides real-time streaming of detection results.
- Allows users to **pause and resume** detection.

## Tech Stack
- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Deep Learning Framework**: TensorFlow, Keras
- **Computer Vision**: OpenCV
- **Models**: CNN for emotion detection, Caffe models for age and gender detection

## Project Structure
```
Facial_1/
│── haarcascades/
│   ├── haarcascade_frontalface_default.xml
│
│── models/
│   ├── age_deploy.prototxt
│   ├── age_net.caffemodel
│   ├── emotion_model.json
│   ├── emotion_model.weights.h5
│   ├── gender_deploy.prototxt
│   ├── gender_net.caffemodel
│
│── static/css/
│   ├── pauseimage.png
│   ├── style.css
│
│── templates/
│   ├── index.html
│
│── uploads/  # Directory for uploaded videos
│── app.py    # Main Flask application
│── README.md
│── requirements.txt
```

## Installation & Setup
### Prerequisites
- **Python 3.10+**

### Running the Application
1. Clone this repository:
    ```sh
    git clone https://github.com/udaypachigolla999/A-DEEP-LEARNING-APPROACH-FOR-RECOGNIZING-AGE-EMOTION-AND-GENDER-IN-FACIAL-EXPRESSIONS.git
    cd A-DEEP-LEARNING-APPROACH-FOR-RECOGNIZING-AGE-EMOTION-AND-GENDER-IN-FACIAL-EXPRESSIONS
    ```
2. Install dependencies using:

    ```sh
    pip install -r requirements.txt
    ```
3. Run the Flask server:
    ```sh
    python app.py
    ```
4. Open a web browser and navigate to:
    ```
    http://127.0.0.1:5000/
    ```
5. Upload a video and start facial recognition!

## Usage
- Click **Upload Video** and select a `.mp4` or `.avi` file.
- Click **Upload and Detect Emotions** to start processing.
- Click **Pause** to stop detection temporarily.
- Click **Play** to resume detection.

## Future Enhancements
- Support for real-time webcam-based detection.
- Extend emotion categories using advanced deep learning models.
- Enhance the UI/UX for a more interactive experience.

## Contributors
- [**Uday Pachigolla**](https://github.com/udaypachigolla999/)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.