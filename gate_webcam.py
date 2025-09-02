import cv2
import os
import json
import numpy as np
from deepface import DeepFace

RESIDENTS_JSON = "moradores.json"

with open(RESIDENTS_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)
    residents = data["moradores"] if "moradores" in data else data

# Pré-calcula embeddings
for r in residents:
    if os.path.isfile(r["rosto"]) and r.get("autorizado", False):
        r["embedding"] = np.array(DeepFace.represent(r["rosto"], enforce_detection=False, detector_backend="opencv")[0]["embedding"])
    else:
        r["embedding"] = None

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_small = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    access_granted = False

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face_roi = cv2.resize(frame_small[y:y+h, x:x+w], (160, 160))
            try:
                frame_embedding = np.array(DeepFace.represent(face_roi, enforce_detection=False, detector_backend="opencv")[0]["embedding"])
                for r in residents:
                    if r["embedding"] is None:
                        continue
                    distance = np.linalg.norm(frame_embedding - r["embedding"])
                    if distance < 0.6:
                        cv2.putText(frame_small, f"Acesso liberado: {r['nome']}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
                        print(f"✅ Acesso liberado para {r['nome']}")
                        access_granted = True
                        break
                if access_granted:
                    break
            except:
                continue

    if not access_granted:
        cv2.putText(frame_small, "Nao Autorizado", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

    cv2.imshow("Camera", frame_small)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
