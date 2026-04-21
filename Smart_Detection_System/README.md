# Smart Detection using face recognition

This project is a smart detection system built using **Python, OpenCV, and face-recognition library**.  
It detects faces using a webcam and automatically marks attendance in a CSV file.

## Features

- Real-time face detection using webcam
- Face recognition using encoding
- Automatic detection marking with time
- CSV attendance record generation
- Fast and efficient recognition

## Technologies Used

- Python
- OpenCV
- face-recognition
- NumPy

## Project Structure
```
Smart_Detection_System
│
├── lost_people_images
│   ├── person1.jpg
│   ├── person2.jpg
│
├── found_people_report.csv
├── finding_people_main.py
├── requirements.txt
└── README.md
```

## How It Works

1. Images of known people are stored inside `lost_people_images`.
2. The system encodes faces from these images.
3. The webcam captures live video.
4. Detected faces are compared with stored encodings.
5. If a match is found, attendance is recorded in `found_people_report`.

