from flask import Flask
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

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)


UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route("/", methods=['GET', 'POST'])
def hello_world():
    if request.method == "POST":
        
        
        file = request.files['image']

        
        # file.save(secure_filename(file.filename))

        pil_image = Image.open(file)
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

if __name__ == "__main__":
    app.run(debug = True)  