from flask import Flask, render_template, request, redirect, send_file, url_for
import cv2
import face_recognition
import numpy as np
import pickle
import os
import pandas as pd
from datetime import datetime
from werkzeug.utils import secure_filename
from io import BytesIO

app = Flask(__name__)

def load_known_encodings(file_path):
    with open(file_path, 'rb') as file:
        known_face_encodings, known_face_names = pickle.load(file)
    return known_face_encodings, known_face_names

def load_and_convert_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, rgb_image

def detect_faces_haar(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
    return faces

def compute_face_encodings(rgb_image, face_locations):
    return face_recognition.face_encodings(rgb_image, face_locations)

def compare_faces(face_encodings, known_face_encodings, known_face_names):
    recognized_students = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        name = "Unknown"
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        recognized_students.append(name)
    return recognized_students

def mark_attendance_in_excel(recognized_students):
    current_date = datetime.now().date()
    current_time = datetime.now().time()
    data = {
        "Student ID": recognized_students,
        "Status": ["Present" if student != "Unknown" else "Absent" for student in recognized_students],
        "Date": [current_date.strftime("%Y-%m-%d")] * len(recognized_students),
        "Time": [current_time.strftime("%H:%M:%S")] * len(recognized_students)
    }
    df = pd.DataFrame(data)

    excel_file = BytesIO()
    with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name="Attendance")
    excel_file.seek(0) 
    return excel_file

def process_and_draw_faces(image_path, recognized_students):
    image, rgb_image = load_and_convert_image(image_path)

    face_locations = detect_faces_haar(image)
    face_locations = sorted(face_locations, key=lambda loc: loc[0]) 
    face_recognition_locations = [(y, x + w, y + h, x) for (x, y, w, h) in face_locations]
    face_encodings = compute_face_encodings(rgb_image, face_recognition_locations)

    recognized_students = compare_faces(face_encodings, recognized_students[0], recognized_students[1])

    for (x, y, w, h), name in zip(face_locations, recognized_students):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    processed_image_path = 'static/processed_image.png'
    cv2.imwrite(processed_image_path, image)
    return processed_image_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    upload_folder = 'static/uploaded_images'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    filename = secure_filename(file.filename)
    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)

    known_face_encodings, known_face_names = load_known_encodings('encodeFile.p')

    try:
        processed_image_path = process_and_draw_faces(file_path, [known_face_encodings, known_face_names])

        # Generate attendance Excel file
        excel_file = mark_attendance_in_excel(known_face_names)

        return render_template('attendance.html', 
                               recognized_students=known_face_names,
                               processed_image_url=url_for('static', filename='processed_image.png'),
                               excel_url=url_for('download_excel'))

    except Exception as e:
        return f"An error occurred: {e}"

@app.route('/download_excel')
def download_excel():
    try:
        known_face_encodings, known_face_names = load_known_encodings('encodeFile.p')

        excel_file = mark_attendance_in_excel(known_face_names)
        return send_file(
            excel_file,
            as_attachment=True,
            download_name="attendance.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        return f"An error occurred while generating the Excel file: {e}"

if __name__ == '__main__':
    app.run(debug=True)
