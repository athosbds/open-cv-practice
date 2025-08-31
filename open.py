from flask import Flask, Response
import cv2
from pyzbar import pyzbar

app = Flask(__name__)
autorizados = ["CARRO123", "VEICULO456"]

cap = cv2.VideoCapture(0)

def gerar_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        codigos = pyzbar.decode(frame)
        for codigo in codigos:
            texto = codigo.data.decode("utf-8").upper()
            (x, y, w, h) = codigo.rect
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if texto in autorizados:
                cv2.putText(frame, "AUTORIZADO", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print(f"Placa {texto} autorizada! Abrindo portão...")
            else:
                cv2.putText(frame, "NAO AUTORIZADO", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(f"Placa {texto} não autorizada.")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def video():
    return Response(gerar_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
