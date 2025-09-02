import cv2
import os
import json
from deepface import DeepFace

RESIDENTS_JSON = "moradores.json"

with open(RESIDENTS_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)
    if isinstance(data, dict) and "moradores" in data:
        residents = data["moradores"]
    elif isinstance(data, list):
        residents = data
    else:
        raise ValueError("Formato de JSON inválido")

for r in residents:
    if not os.path.isfile(r["rosto"]):
        print(f"Imagem não encontrada para {r['nome']}: {r['rosto']}")
        r["autorizado"] = False

cap = cv2.VideoCapture(0)
print("Pressione 'q' para sair.")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Não foi possível capturar o frame da câmera.")
        break

    frame_small = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    access_granted = False

    if len(faces) > 0:
        for r in residents:
            if not r.get("autorizado", False):
                continue
            try:
                result = DeepFace.verify(
                    frame_small,
                    r["rosto"],
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                if result["verified"]:
                    cv2.putText(frame_small,
                                f"Acesso liberado: {r['nome']}",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                2,
                                cv2.LINE_AA)
                    print(f"✅ Acesso liberado para {r['nome']}")
                    print("🔓 Portão aberto - Simulação")
                    access_granted = True
                    break
            except Exception as e:
                print(f"Nenhuma face detectada ou erro: {e}")
    else:
        cv2.putText(frame_small,
                    "Nao Autorizado",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA)

    cv2.imshow("Camera", frame_small)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
