from PIL import Image
import cv2
import face_recognition
from IPython.display import display, clear_output

# Load the image containing the reference face
reference_image = face_recognition.load_image_file(r"D:\Pan\pandata\raja.jpg")

# Extract the face encoding from the reference image
reference_encoding = face_recognition.face_encodings(reference_image)[0]

# Open the video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = video_capture.read()

    # Convert the frame to RGB format
    rgb_frame = frame[:, :, ::-1]

    # Find all the face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame, model='cnn')
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Iterate over the face encodings found in the current frame
    for face_encoding in face_encodings:
        # Compare the face encoding with the reference encoding
        match_score = face_recognition.compare_faces([reference_encoding], face_encoding)[0]
        
        match_percentage = face_recognition.face_distance([reference_encoding], face_encoding)
        match_percentage = (1 - match_percentage[0]) * 100

        # Print the match score
        print("Match score:", match_score, match_percentage)

    # Display the frame with bounding boxes around the detected faces
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Convert the frame from OpenCV BGR to PIL RGB format for Jupyter Notebook display
    pil_frame = Image.fromarray(frame)

    # Display the frame in Jupyter Notebook
    display(pil_frame)
    clear_output(wait=True)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
