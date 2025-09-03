import cv2
import json
import os
import face_recognition

RESIDENTS_JSON = "moradores.json"

# carregar moradores
with open(RESIDENTS_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)
    residents = data.get("moradores", data)


known_face_encodings = []
known_face_names = []
authorized_status = []

for resident in residents:
    if "nome" in resident and "rosto" in resident:
        archive = resident["rosto"]
        if os.path.exists(archive):
            image = face_recognition.load_image_file(archive)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(resident["nome"])
                authorized_status.append(resident.get("autorizado", False))
            else:
                print(f"[AVISO] Nenhum rosto detectado: {archive}")
        else:
            print(f"[ERRO] Arquivo não encontrado: {archive}")


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

 
    small_frame = cv2.resize(frame, (120, 90))
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)


    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding, tolerance=0.6
        )
        name = "Desconhecido"
        authorized = False

        if True in matches:
            idx = matches.index(True)
            name = known_face_names[idx]
            authorized = authorized_status[idx]

            if authorized:
                print(f"✅ Portão aberto para: {name}")
            else:
                print(f"❌ {name} reconhecido, mas sem autorização")
        else:
            print("⚠️ Rosto desconhecido detectado")

        top, right, bottom, left = [v * int(frame.shape[0] / small_frame.shape[0]) for v in face_location]
        color = (0, 255, 0) if authorized else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Reconhecimento Facial", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
