import cv2
import json
import os
from deepface import DeepFace
import numpy as np

with open("moradores.json", "r", encoding="utf-8") as f:
    residents = json.load(f)["moradores"]


for resident in residents:
    if not os.path.isfile(resident["rosto"]):
        print(f"Imagem do morador {resident['nome']} não encontrada: {resident['rosto']}")


cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
print("Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível capturar o frame da câmera.")
        break
    cv2.imshow("Camera", frame)
    for resident in residents:
        try:
            if os.path.isfile(resident["rosto"]):
                resultado = DeepFace.verify(
                    frame,
                    resident["rosto"],
                    enforce_detection=False
                )
                if resultado["verified"] and resident["autorizado"]:
                    print(f"Acesso liberado para {resident['nome']}")
                    print("Portão aberto - Simulando")
                    break
        except Exception as e:
            print(f"erro na verificação do morador {resident['nome']}: {e}")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
