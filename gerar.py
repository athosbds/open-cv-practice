import qrcode

placa = "CARRO123"

qr = qrcode.QRCode(version=1, box_size=10, border=4)
qr.add_data(placa)
qr.make(fit=True)

img = qr.make_image(fill_color="black", back_color="white")
img.save("carro123.png")
