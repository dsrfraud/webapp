from flask import Flask, Response, jsonify, request, render_template


from PIL import Image
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import os
from werkzeug.utils import secure_filename
import base64
import yolo_prediction
import cv2
import numpy as np
import io
import json

import speech_recognition as sr

import face_recognition
import mediapipe as mp

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)


UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER




mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def generate_frames(cap):
    
    while cap.isOpened():
        success, image = cap.read()
        # Flip the image horizontally for a later selfie-view display
        # Also convert the color space from BGR to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance
        image.flags.writeable = False

        # Get the results
        results = face_mesh.process(image)

        # To improve performance
        image.flags.writeable = True

        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])       

                face_2d = np.array(face_2d, dtype=np.float64)

                face_3d = np.array(face_3d, dtype=np.float64)
                
                        # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])
                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                
                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)
                
                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360

                # print(y)

                # See where the user's head tilting
                if y < -10:
                    text = "Looking Left"
                    print("Looking Left, X = {0},Y={1}".format(x,y))
                elif y > 15:
                    text = "Looking Right"
                    print("Looking Right, X = {0},Y={1}".format(x,y))
                elif x < -10:
                    text = "Looking Down"
                    print("Looking Down, X = {0},Y={1}".format(x,y))
                elif ((x > 0) & (x <15)):
                    text = "Forward"
                    print("Forward, X = {0},Y={1}".format(x,y))
                elif x>10:
                    text = "Looking Up"
                    print("Looking Up, X = {0},Y={1}".format(x,y))
                
                # Add the text on the image
                cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
                
                cv2.line(image, p1, p2, (255, 0, 0), 2)
        ret,buffer=cv2.imencode('.jpg',image)
        frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



def get_word():
    words = ['apple', 'banana', 'cherry', 'date', 'elderberry', 'fig', 'grape', 'honeydew', 'indigo', 'kiwi', 'lemon', 'mango', 
    'orange', 'peach', 'raspberry', 'strawberry', 'ugli', 'vanilla', 'watermelon', 'yellow', 'zebra']

    np.random.shuffle(words)

    return words[:3]


def check_position(list1, list2):

    mismatched_list = []
    matched_list = []
    for i in range(len(list2)):
        try:
            if list2[i].lower() != list1[i].lower():
                mismatched_list.append(list2[i])
            

            
        except Exception as e:
            pass
    return mismatched_list


@app.route('/facematch', methods = ['POST', "GET"])
def facematch():
    if request.method == "POST":
        print(f'&*ERDF')
        # Get the uploaded image file
        uploaded_image = request.files['uploadedImage']
        # Read the uploaded image using OpenCV
        uploaded_image_cv2 = cv2.imdecode(np.fromfile(uploaded_image, np.uint8), cv2.IMREAD_COLOR)

        # Get the captured image as a base64-encoded string
        captured_image_base64 = request.form['capturedImage']
        # Convert the base64-encoded captured image to a numpy array
        captured_image_data = np.frombuffer(base64.b64decode(captured_image_base64), np.uint8)
        captured_image_cv2 = cv2.imdecode(captured_image_data, cv2.IMREAD_COLOR)

        skew = yolo_prediction.find_skew_angle(uploaded_image_cv2)
        print(f'Skewness {skew}')
        # print(f'SKEW ANgle {skew}')

        rotated_image = yolo_prediction.rotate_image(uploaded_image_cv2, skew) 
        # print("Roated")
        predictions = yolo_prediction.get_prediction(rotated_image)
        # print(predictions)
        # Get the face coordinates for the photo
        face_coordinates = predictions.loc[predictions['name'] == 'photo', ['xmin', 'ymin', 'xmax', 'ymax']].values[0]

        # Convert coordinates to integers
        xmin = int(face_coordinates[0])
        ymin = int(face_coordinates[1])
        xmax = int(face_coordinates[2])
        ymax = int(face_coordinates[3])

        # Crop the photo from the image
        cropped_photo = uploaded_image_cv2[ymin:ymax, xmin:xmax]

        # Convert the BGR images to RGB format
        captured_image_rgb = cv2.cvtColor(cropped_photo, cv2.COLOR_BGR2RGB)
        uploaded_image_rgb = cv2.cvtColor(captured_image_cv2, cv2.COLOR_BGR2RGB)

        cv2.imwrite("captured.jpg", captured_image_rgb)
        cv2.imwrite("uploaded_image_rgb.jpg", uploaded_image_rgb)

        # Find face locations and encodings in the images
        captured_face_locations = face_recognition.face_locations(captured_image_rgb)
        captured_face_encodings = face_recognition.face_encodings(captured_image_rgb, captured_face_locations)
        uploaded_face_locations = face_recognition.face_locations(uploaded_image_rgb)
        uploaded_face_encodings = face_recognition.face_encodings(uploaded_image_rgb, uploaded_face_locations)

        # Check if at least one face is found in each image
        if len(captured_face_encodings) > 0 and len(uploaded_face_encodings) > 0:
            # Take the first face from each image
            captured_encoding = captured_face_encodings[0]
            uploaded_encoding = uploaded_face_encodings[0]

            # Compare the face encodings
            match_percentage = face_recognition.face_distance([captured_encoding], uploaded_encoding)
            match_percentage = (1 - match_percentage[0]) * 100

            # Print the match percentage
            print("Match percentage:", match_percentage)
        else:
            print("No face found in one or both of the images.")

        if match_percentage:
            data = {
                "Matched Score" : int(match_percentage)
            }
            return jsonify(data)

        

        # print(uploaded_image_cv2, captured_image_cv2)
    return render_template('facematch.html')


@app.route('/audio', methods = ['POST', 'GET'])
def audio():
    words = get_word()
    words = ",".join(word for word in words)
    words = json.dumps(words)
    if request.method == "POST":
        if 'audio' not in request.files:
            return 'No audio file found', 400
    
        audio_file = request.files['audio']
        received_words = request.form.get('words')
        recognizer = sr.Recognizer()

        # Read the audio file
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)

        # Perform speech recognition
        try:
            text = recognizer.recognize_google(audio)
            text = text.split(" ")
            mismatched_list = check_position(received_words.split(","), text)
            print(f'Mismatched Words {mismatched_list}')
            
            data = {"spoke" : text, "mismatched" : mismatched_list}
            return jsonify(data)
        except sr.UnknownValueError:
            return {'error': 'Speech recognition could not understand audio'}
        except sr.RequestError as e:
            return {'error': f'Speech recognition service error: {str(e)}'}
        
    
    return render_template('audio.html', words = words)

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/videodemo')
def videoDe():
    directions = np.array([['Forward', 'Looking Right', 'Forward', 'Looking Left', 'Forward', 'Looking Up','Forward', 'Looking Down'], ['Forward', 'Looking Left', 'Forward',
    'Looking Right', 'Forward', 'Looking Up', 'Forward', 'Looking Down']])
    np.random.shuffle(directions)
    return render_template('video2.html', directions = json.dumps(directions[0].tolist()))

@app.route("/", methods=['GET', "POST"])
def home():
    return render_template('home.html')

@app.route("/detection", methods=['GET', 'POST'])
def hello_world():
    if request.method == "POST":
        
        
        file = request.files['image']
        pil_image = Image.open(file)
        data = io.BytesIO()
        pil_image.save(data, "JPEG")
        encoded_img_data = base64.b64encode(data.getvalue())

        opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        skew = yolo_prediction.find_skew_angle(opencvImage)
        # print(f'SKEW ANgle {skew}')

        rotated_image = yolo_prediction.rotate_image(opencvImage, skew) 
        # print("Roated")
        predictions = yolo_prediction.get_prediction(rotated_image)
        result  = yolo_prediction.get_ocr(rotated_image, predictions)
        return render_template('index.html', filename=encoded_img_data.decode('utf-8'),result = result) 
   
    return render_template('index.html')


import uuid
@app.route('/upload', methods=['POST'])
def upload():
    video = request.files['video']
    directions = request.form.get('direction')
    directions = directions.split(",")
    filename = str(uuid.uuid4())+"webm"
    video.save('/home/ubuntu/webapp/static/uploads/'+filename) 
    # Create a VideoCapture object to read the video
    cap = cv2.VideoCapture('/home/ubuntu/webapp/static/uploads/'+filename)
    result_list  = []
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        # Flip the image horizontally for a later selfie-view display
        # Also convert the color space from BGR to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance
        image.flags.writeable = False

        # Get the results
        results = face_mesh.process(image)

        # To improve performance
        image.flags.writeable = True

        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])       

                face_2d = np.array(face_2d, dtype=np.float64)

                face_3d = np.array(face_3d, dtype=np.float64)
                
                        # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])
                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                
                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)
                
                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360

                # print(y)

                # See where the user's head tilting
                if y < -10:
                    text = "Looking Left"
                    # print("Looking Left, X = {0},Y={1}".format(x,y))
                elif y > 15:
                    text = "Looking Right"
                    # print("Looking Right, X = {0},Y={1}".format(x,y))
                elif x < -10:
                    text = "Looking Down"
                    # print("Looking Down, X = {0},Y={1}".format(x,y))
                elif ((x > 0) & (x <15)):
                    text = "Forward"
                    # print("Forward, X = {0},Y={1}".format(x,y))
                elif x>10:
                    text = "Looking Up"
                    # print("Looking Up, X = {0},Y={1}".format(x,y))
                try:
                    result_list.append(text)
                except:
                    result_list.append("")
                
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
                
                # cv2.line(image, p1, p2, (255, 0, 0), 2)
    # print(result_list)
    seen_directions = []
    seen_directions.append(result_list[0])
    for i,a in enumerate(result_list):
        
        if (i > 0) & (result_list[i] != result_list[i-1]):
            seen_directions.append(a)
    seen_directions = [i for i in seen_directions if i !=""]
    print(f'Directions Detected : {seen_directions}')
    
    print(f'Directions Coming : {directions}')
    # result_list = ", ".join(x for x in result_list)
    if seen_directions == directions:
        return "Successfully Matched"
    
    else:
        return "Not Matched" 

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug = True)
