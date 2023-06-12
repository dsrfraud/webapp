from flask import Flask, Response, jsonify
from flask import render_template
from flask import request
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


import base64 as b64
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

# cap=cv2.VideoCapture(0)

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

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/videodemo')
def videoDe():
    print(f"*"*20)
    
    return render_template('video2.html')

@app.route("/", methods=['GET', "POST"])
def home():
    return render_template('home.html')

@app.route("/detection", methods=['GET', 'POST'])
def hello_world():
    if request.method == "POST":
        
        
        file = request.files['image']

        
        # file.save(secure_filename(file.filename))

        pil_image = Image.open(file)
        # # Set the new size of the image
        # new_size = (512, 512)

        # # Resize the image using Pillow
        # pil_image = pil_image.resize(new_size)
        data = io.BytesIO()

        #First save image as in-memory.
        pil_image.save(data, "JPEG")

        #Then encode the saved image file.
        encoded_img_data = base64.b64encode(data.getvalue())

        opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        # filename = file.filename
        
        
        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # filepath  = 'http://127.0.0.1:5000/static/uploads/'  + filename

        skew = yolo_prediction.find_skew_angle(opencvImage)
        # print(f'SKEW ANgle {skew}')

        rotated_image = yolo_prediction.rotate_image(opencvImage, skew) 
        # print("Roated")
        predictions = yolo_prediction.get_prediction(rotated_image)
        result  = yolo_prediction.get_ocr(rotated_image, predictions)
        print(result)
        return render_template('index.html', filename=encoded_img_data.decode('utf-8'),result = result) 
   
    return render_template('index.html')


import uuid
@app.route('/upload', methods=['POST'])
def upload():
    video = request.files['video']
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
                try:
                    result_list.append(text)
                except:
                    result_list.append("")
                # Add the text on the image
                # cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
                
                # cv2.line(image, p1, p2, (255, 0, 0), 2)
    print(result_list)
    result_list = ", ".join(x for x in result_list)

    return result_list

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug = True)
