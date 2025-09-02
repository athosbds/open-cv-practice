import cv2
import json
import os
import numpy as np
from deepface import DeepFace

# -------------------------
# Arquivos
# -------------------------
RESIDENTS_JSON = "moradores.json"
EMBEDDINGS_BACKUP = "embeddings_backup.json"

# -------------------------
# Carregar moradores
# -------------------------
with open(RESIDENTS_JSON, "r", encoding="utf-8") as f:
    residents = json.load(f)["moradores"]

# -------------------------
# Pré-calcular embeddings se backup não existir
# -------------------------
if os.path.isfile(EMBEDDINGS_BACKUP):
    with open(EMBEDDINGS_BACKUP, "r", encoding="utf-8") as f:
        residents = json.load(f)["moradores"]
    print("✅ Embeddings carregados do backup.")
else:
    for r in residents:
        if os.path.isfile(r["rosto"]):
            try:
                r["embedding"] = DeepFace.represent(r["rosto"], enforce_detection=False, detector_backend='opencv')[0]["embedding"]
            except Exception as e:
                print(f"⚠️ Erro ao gerar embedding para {r['nome']}: {e}")
                r["embedding"] = None
        else:
            print(f"⚠️ Imagem não encontrada para {r['nome']}: {r['rosto']}")
            r["embedding"] = None

    with open(EMBEDDINGS_BACKUP, "w", encoding="utf-8") as f:
        json.dump(residents, f)
    print("✅ Embeddings salvos no backup.")

# -------------------------
# Configurar câmera
# -------------------------
cap = cv2.VideoCapture(0)
frame_count = 0
print("Pressione 'q' para sair.")


while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Não foi possível capturar o frame da câmera.")
        break


    frame_small = cv2.resize(frame, (320, 240))
    cv2.imshow("Camera", frame_small)
    frame_count += 1


    if frame_count % 5 == 0:
        try:
            frame_embedding = DeepFace.represent(frame_small, enforce_detection=False, detector_backend='opencv')[0]["embedding"]
            for r in residents:
                if r.get("embedding") is not None:
                    distance = np.linalg.norm(np.array(frame_embedding) - np.array(r["embedding"]))
                    if distance < 0.6 and r["autorizado"]:
                        print(f"✅ Acesso liberado para {r['nome']}")
                        print("🔓 Portão aberto - Simulação")
                        break
        except Exception as e:
            print(f"⚠️ Erro na verificação do frame: {e}")

    # Sair com 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
