import cv2
import json
import os
import face_recognition
import numpy as np

RESIDENTS_JSON = "moradores.json"

# carregar moradores
with open(RESIDENTS_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)
    residents = data.get("moradores", data)


known_face_encodings = []
known_face_names = []
authorized_status = []

for resident in residents:
    if "nome" in resident:
        face_files = resident.get("rostos", [])
        if isinstance(face_files, str):
            face_files = [face_files]
        for archive in face_files:
            if not os.path.exists(archive):
                print(f'Arquivo Não Encontrado.')
                continue
            image = face_recognition.load_image_file(archive)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(resident["nome"])
                authorized_status.append(resident.get("autorizado", False))
            else:
                print(f"[AVISO] Nenhum rosto detectado: {archive}")


cap = cv2.VideoCapture(0)
print("'q' para sair.")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 15 != 0:
        cv2.imshow("Reconhecimento Facial", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        name = "Desconhecido"
        authorized = False

        if known_face_encodings:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if face_distances[best_match_index] < 0.7:
                name = known_face_names[best_match_index]
                authorized = authorized_status[best_match_index]

        print(f"{'✅' if authorized else '❌'} {name}")

        # desenhar retângulo
        top, right, bottom, left = [v * 2 for v in face_location]
        color = (0, 255, 0) if authorized else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("Reconhecimento Facial", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()