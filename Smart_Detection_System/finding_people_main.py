import cv2
import numpy as np
import os
import face_recognition
from datetime import datetime




path = 'lost_people_images'
images = []
class_names = []
my_list = os.listdir(path)
print(my_list)

for cl in my_list:
    curr_image = cv2.imread(f'{path}/{cl}')
    images.append(curr_image)
    class_names.append(os.path.splitext(cl)[0])

print(class_names)




def find_encodings(images):
    encoding_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encoding_list.append(encode)
        
    return encoding_list




def mark_attendance(name):
    with open("found_people_report", 'r+') as f:
        my_data_lsit = f.readlines()
        name_list = []
        
        for line in my_data_lsit:
            entry = line.split(",")
            name_list.append(entry[0])
        
        if name not in name_list:
            now = datetime.now()
            date_string = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{date_string}')
            
    



encoding_list_known = find_encodings(images)
print("Encoding Completed ", len(encoding_list_known))




cap = cv2.VideoCapture(0)

while True:
    flag, img = cap.read()
    img_smaller = cv2.resize(img, (0,0), None, 0.25, 0.25)
    img_smaller = cv2.cvtColor(img_smaller, cv2.COLOR_BGR2RGB)
    
    faces_curr_frame = face_recognition.face_locations(img_smaller)
    encodes_curr_frame = face_recognition.face_encodings(img_smaller, faces_curr_frame)
    
    for encode_face, face_location in zip(encodes_curr_frame, faces_curr_frame):
        matches = face_recognition.compare_faces(encoding_list_known, encode_face)
        face_distance = face_recognition.face_distance(encoding_list_known, encode_face)
        print(face_distance)
        
        matched_index = np.argmin(face_distance)
        
        if matches[matched_index]:
            name = class_names[matched_index].upper()
            print(name)
            y1, x2, y2, x1 = face_location
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            mark_attendance(name)
            
            
            
    cv2.imshow("Webcam", img)
    if cv2.waitKey(10) == 27:
        break
    



cap.release()
cv2.destroyAllWindows()    
