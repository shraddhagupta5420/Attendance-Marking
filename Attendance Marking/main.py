# Importing All The Necessary Libraries.
import cv2
import dlib
from datetime import datetime, timedelta
import os
import numpy as np
import pickle
import time
import pandas as pd
import streamlit as st

# Add custom CSS styles to improve the appearance of the app
st.markdown(
    """
    <style>
        /* Center the title */
        .title {
            text-align: center;
            font-size: 2.5em;
            color: #3366ff;
            margin-bottom: 1em;
        }

        /* Add padding to the main content */
        .content {
            padding: 2em;
        }

        /* Style the sidebar */
        .sidebar .sidebar-content {
            background-color: #f0f0f0;
            padding: 1em;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        /* Style the sidebar header */
        .sidebar .sidebar-content .sidebar-header {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 0.5em;
        }

        /* Style the image container */
        .image-container {
            text-align: center;
            margin-bottom: 2em;
        }

        /* Style the image */
        .student-image {
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        /* Style the success message */
        .success-message {
            text-align: center;
            color: #008000;
            font-weight: bold;
            margin-top: 1em;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("Attendance Marking System")

# Sidebar
st.sidebar.header("Options")
# Placeholder for success message
success_message_placeholder = st.empty()

cap = cv2.VideoCapture(0)

# The Folder Where My Dataset Is Stored.
path = "StudentImages"

''' 
Here I am Creating The 2 Empty Lists In 
Which I am Storing Images in One list And
In Another I am Storing Name By Using For Loop.
'''

images = []
classname = []
myList = os.listdir(path)

for cl in myList:
    # Reading The Images From The Dataset.
    curImg = cv2.imread(f"{path}/{cl}")
    # Then Append Images In The List Named 'images'.
    images.append(curImg)
    # Here I Am Appending The Name Of Images In List Named 'classname'.
    classname.append(os.path.splitext(cl)[0])

# Use The dlib Library For Face Recognition.
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load The Pre-Trained Face Recognition Model.
face_rec_model = dlib.face_recognition_model_v1(
    "dlib_face_recognition_resnet_model_v1.dat")

'''
Here I Am Checking If 'encoding.pkl' File Exists And 
If It Exists Then It Will Check Face Encoding From That
File And Not Generate Encoding Of Dataset Again So It
Will Execute Faster This Way.
'''

# Storing File Name In Variable.
encodings_file = 'encodings.pkl'
# Checking  If That File Exists Or Not.
if os.path.exists(encodings_file):
    # If That File Exists Opening That File In ReadBinary Mode.
    with open(encodings_file, 'rb') as f:
        # Then Here I Am Unpickeling That File And Store Content Of That File In Variable.
        encoded_faces_train = pickle.load(f)
else:
    # Encoding faces outside the loop using dlib
    # Creating The Empty List For Storing The Encodings Of Images Outside The Loop.
    encoded_faces_train = []
    # Here I Am Reading The Images From The List Of Stored Images.
    for img in images:
        # And The Converting That All Images From BGR 2 RGB (Red, Green, Blue) .
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Using The face_detector Function I am Detecting The Face Location.
        face_locations = face_detector(img_rgb, 1)
        for face_location in face_locations:
            shape = shape_predictor(img_rgb, face_location)

        face_encoding = face_rec_model.compute_face_descriptor(img_rgb, shape)
        encoded_faces_train.append(face_encoding)

    '''
    In Above Function I Am storing The Encoding Of Photos If That
    'encoding.pkl' File Does Not Exist It Will Create The New File 
    And Store The Encodings In That File. 
    '''
    with open(encodings_file, 'wb') as f:
        pickle.dump(encoded_faces_train, f)


def markAttendance(name, session_type):
    # Define the filename for the Excel file based on the session type
    if session_type == "theory":
        excel_file = 'theory_attendance.xlsx'
    else:
        excel_file = 'lab_attendance.xlsx'

    # Get the current date and the previous day's date
    currentDate = datetime.now().strftime('%d-%B-%Y')
    previousDate = (datetime.now() - timedelta(days=1)).strftime('%d-%B-%Y')

    # Check if the Excel file exists
    if os.path.exists(excel_file):
        # Read the existing Excel file
        df = pd.read_excel(excel_file)

        # Check if the person already exists in the DataFrame
        if name in df['Name'].values:
            # If the person exists, update the attendance for the current date
            df.loc[df['Name'] == name, currentDate] = 'P'

            # Check if the person was absent on any previous day(s)
            for date_column in df.columns:
                if date_column != 'Name' and date_column != currentDate:
                    if df.loc[df['Name'] == name, date_column].iloc[0] != 'P':
                        # If the person was absent, mark "A" for them on that day
                        df.loc[df['Name'] == name, date_column] = 'A'
        else:
            # If the person doesn't exist, create a new DataFrame for the new row
            new_row = pd.DataFrame({'Name': [name], currentDate: ['P']})

            # Concatenate the new row DataFrame with the existing DataFrame
            df = pd.concat([df, new_row], ignore_index=True)

        # Write the updated DataFrame back to the Excel file
        df.to_excel(excel_file, index=False)

    else:
        # Create a new DataFrame with the person's name and mark attendance for the current date
        df = pd.DataFrame({'Name': [name], currentDate: ['P']})

        # Write the DataFrame to the Excel file
        df.to_excel(excel_file, index=False)

    # Display success message for marking attendance
    update_success_messages(name)


# Placeholder for the success message
success_message_placeholder = st.empty()

# Placeholder for the image display
img_placeholder = st.empty()

# Initialize variables for success message display
success_names = []
message_displayed = False
message_start_time = None

# Function to update the success messages


def update_success_messages(name):
    if name not in success_names:
        success_names.append(name)


while True:
    success, img = cap.read()
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Use dlib for face detection
    face_locations = face_detector(img_rgb, 1)

    # Use dlib for face recognition
    encoded_faces = [face_rec_model.compute_face_descriptor(
        img_rgb, shape_predictor(img_rgb, location)) for location in face_locations]

    for encoded_face, faceloc in zip(encoded_faces, face_locations):
        matches = [np.linalg.norm(np.array(encoded_face) - np.array(train_face)) < 0.55
                   for train_face in encoded_faces_train]

        if any(matches):
            matchIndex = matches.index(True)
            name = classname[matchIndex]

            # Draw rectangle around the face
            top, right, bottom, left = faceloc.top(
            ), faceloc.right(), faceloc.bottom(), faceloc.left()
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw semi-transparent rectangle above the face
            overlay = img.copy()
            cv2.rectangle(overlay, (left, bottom - 35),
                          (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

            # Draw text above the face
            cv2.putText(img, name, (left + 6, bottom - 5),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Mark attendance for theory session (9 AM - 1 PM) or practical session (2 PM - 6 PM)
            now = datetime.now()
            if now.hour >= 9 and now.hour < 14:
                session_type = "theory"
            else:
                if now.hour >= 14 and now.hour < 18:  # Adjust the time range for practical session
                    session_type = "practical"
                else:
                    session_type = "other"  # Handle other cases if needed
    
            # Check if attendance is open for the current session type
            if session_type != "other":
                markAttendance(name, session_type)
            else:
                # Display message when attendance is closed
                st.warning("Attendance is closed for the current session.")
                update_success_messages(name)

    # Display video frame in Streamlit
    img_placeholder.image(img, channels="BGR", use_column_width=True)

    # Check if the attendance is closed for the current session
    attendance_closed = False
    now = datetime.now()
    if now.hour < 9 or now.hour >= 14:
        attendance_closed = True

    # Display success messages with late message if applicable
    if success_names and not message_displayed:
        message_displayed = True
        message_start_time = time.time()
        # Check if the session is closed
        now = datetime.now()
        if (now.hour >= 9 and now.hour < 14) or (now.hour >= 14 and now.hour < 18):
            # Session is open, display success messages
            # Calculate late message if applicable
            expected_time = datetime(now.year, now.month, now.day, 9, 0) if now.hour < 14 else datetime(now.year, now.month, now.day, 14, 0)
            late_time = now - expected_time
            late_minutes = late_time.total_seconds() // 60
            late_message = f"Late by {int(late_minutes)} minutes." if late_minutes > 10 else ""
            # Display success message with late message if applicable
            success_message = "\n".join(
                f"{name} has successfully marked attendance. {late_message}" for name in success_names)
            success_message_placeholder.markdown(
                "<div class='success-message'>{}</div>".format(success_message), unsafe_allow_html=True)
            

        else:
            # Session is closed, display warning message
            success_message_placeholder.warning("Attendance is closed for the current session.")



    # Clear success names after displaying for 2 seconds
    if message_displayed and time.time() - message_start_time >= 2:
        success_names.clear()
        message_displayed = False

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

