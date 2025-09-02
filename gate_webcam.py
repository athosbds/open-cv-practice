import cv2
import os
import json
import numpy as np
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
        print(f"⚠️ Imagem não encontrada para {r['nome']}: {r['rosto']}")
        r["embedding"] = None
    elif not r.get("autorizado", False):
        r["embedding"] = None
    else:
        try:
            r["embedding"] = DeepFace.represent(
                r["rosto"],
                enforce_detection=False,
                detector_backend='opencv'
            )[0]["embedding"]
        except Exception as e:
            print(f"⚠️ Erro ao gerar embedding para {r['nome']}: {e}")
            r["embedding"] = None


cap = cv2.VideoCapture(0)
frame_count = 0
print("Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Não foi possível capturar o frame da câmera.")
        break

    frame_small = cv2.resize(frame, (640, 480))
    frame_count += 1


    if frame_count % 5 == 0:
        try:
            frame_embedding = DeepFace.represent(
                frame_small,
                enforce_detection=False,
                detector_backend='opencv'
            )[0]["embedding"]

            access_granted = False
            for r in residents:
                if r.get("embedding") is None:
                    continue
 
                distance = np.linalg.norm(np.array(frame_embedding) - np.array(r["embedding"]))
                print(f"Distância para {r['nome']}: {distance:.3f}")
                if distance < 0.6:
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

            if not access_granted:
                cv2.putText(frame_small,
                            "Nao autorizado",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA)

        except Exception as e:
            print(f"⚠️ Nenhuma face detectada ou erro: {e}")


    cv2.imshow("Camera", frame_small)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.d
