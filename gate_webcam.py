import cv2
import json
import os
from deepface import DeepFace

with open("moradores.json", "r", encoding="utf-8") as f:
    residents = json.load(f)["moradores"]


for resident in residents:
    if not os.path.isfile(resident["rosto"]):
        print(f"Imagem do morador {resident['nome']} não encontrada: {resident['rosto']}")


cap = cv2.VideoCapture(0)
print("q para sair")
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("captura de câmera negada")
        break
    frame_small = cv2.resize(frame, (320, 240))
    cv2.imshow("Camera", frame_small)
    frame_count += 1
    if frame_count % 5 == 0:
        for resident in residents:
            try:
                if os.path.isfile(resident["rosto"]):
                    resultado = DeepFace.verify(
                    frame_small,
                    resident["rosto"],
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                if resultado["verified"] and resident["autorizado"]:
                    print(f"acesso liberado para {resident['nome']}")
                    print("Portão aberto - Simulando")
                    break
            except Exception as e:
                print(f"erro na verificação do morador {resident['nome']}: {e}")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
