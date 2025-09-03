import cv2
import json
import os
import face_recognition


RESIDENTS_JSON = "moradores.json"

with open(RESIDENTS_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)
    residents = data["moradores"] if "moradores" in data else data
known_face = []
known_face_name = []
authorized_status = []

for resident in residents:
    if "nome" in resident["rosto"] in resident:
        archive = resident["rosto"]
        if os.path.exists(archive):
            image = face_recognition.load_image_file(archive)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face.append(encodings[0])
                known_face_name.append(resident["nome"])
            else:
                print(f"[AVISO] - Nenhum Rosto Detectado: {archive}")
        else:
            print(f'[ERRO] Arquivo não encontrado: {archive}')


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # frames
    small = cv2.resize(frame (320, 240))
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    face_location = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_locations(rgb_small, face_location)
    for face_encoding, face_location in zip(face_encodings, face_location):

        matches = face_recognition.compare_faces(known_face, face_encoding, tolerance=0.5)
        name = "Desconhecido"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_name[first_match_index]
            print(f"✅ Portão aberto para: {name}")

        top, right, bottom, left = [v * 4 for v in face_location]  l
        color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Reconhecimento Facial", frame)

    # Sair no 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()