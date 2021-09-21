from yes import face_analysis
import numpy as np
import cv2
import dlib
import uuid
import face_recognition

from face_recog.face_data_store import FaceDataStore

def encodeimg(img,bbox,crop_forehead):
	x,y,w,h=bbox[0],bbox[1],bbox[2],bbox[3]
	x1,y1=abs(x),abs(y)
	x2,y2=x1+h,y1+w
	if crop_forehead:
		y1=y1+int(h*0.2)
	
	face_encoding = face_recognition.face_encodings(img,model='large',num_jitters=1,known_face_locations=[[y1,x2,y2,x1]])[0]
	face_encoding = np.array(face_encoding)
	return face_encoding



def registerface(image_path,name,crop_forehead=True):
	
	img,bboxes,conf=face_analysis().face_detection(image_path=image_path,model='full')

	bbox = bboxes[0]
	face_encoding = encodeimg(img,bbox,crop_forehead)
	
	facial_data = {"id": str(uuid.uuid4()), "encoding": tuple(face_encoding.tolist()), "name": name}
	FaceDataStore(persistent_data_loc="data/try.json").add_facial_data(facial_data=facial_data)


def recogimg(image_path=None,frame=None,write=False):
	names = []
	if frame is None:
		img,bboxes,conf=face_analysis().face_detection(image_path=image_path,model='full')
	
	if image_path is None:
		img,bboxes,conf=face_analysis().face_detection(frame_arr=frame,frame_status=True,model='full')
	
	all_facial_data=FaceDataStore(persistent_data_loc="data/try.json").get_all_facial_data()
	for bbox in bboxes:
		
		face_encode = encodeimg(img,bbox,crop_forehead=False)
		distances = []
		
		for face_data in all_facial_data:
			dist=np.linalg.norm( np.array(face_encode)/np.linalg.norm(face_encode) - np.array(face_data["encoding"])/np.linalg.norm(face_data["encoding"]) )
			distances.append(dist)

		min_dist = min(distances)
		pos = distances.index(min_dist)
		match=all_facial_data[pos]
		name = match["name"] if match is not None else "Unknown"
		names.append(name)

	return names

def recogvid(video_path,write=False):
	cap = cv2.VideoCapture(video_path)
	frame_num = 1
	all_names = []

	while True:
		status, frame = cap.read()
		if not status:
			break
		image=frame.copy()
		if frame_num % 5 == 0:
			names = recogimg(frame=image)
			all_names.append(names)

		frame_num+=1

	cap.release()
	final_list = [item for sublist in all_names for item in sublist]
	return np.unique(np.array(final_list))


def deletefaces():
    import os
    if os.path.exists('data/try.json'):
        os.remove('data/try.json')
        print('[INFO] Test DB file deleted...')


deletefaces()
registerface('data/sample/chris evans.jpg','Chris Evans')
registerface('data/sample/chris hemsworth.jpg','Chris Hemsworth')
registerface('data/sample/robert.jpg','Robert')
registerface('data/sample/sca.jpg','Sca')
registerface('data/sample/mark.jpg','Mark')
registerface('data/sample/jeremy.jpg','Jeremy')

print(recogimg("together.jpg"))
print(recogvid("try2.mp4"))
