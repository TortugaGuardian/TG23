import cv2
from twilio.rest import Client

# substitua as informações da sua conta Twilio abaixo
account_sid = 'SUA_CONTA_SID'
auth_token = 'SEU_AUTH_TOKEN'
client = Client(account_sid, auth_token)

# carregue o modelo de detecção de objetos treinado
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# inicie a câmera
cap = cv2.VideoCapture(0)

while True:
    # leia o próximo frame da câmera
    ret, frame = cap.read()

    # execute a detecção de objetos no frame
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # verifique se há detecções com bordas visíveis
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            if class_id == 1:  # a classe 1 é "pessoa" neste modelo
                # envie uma mensagem usando o Twilio
                message = client.messages.create(
                    to='SEU_NUMERO_DE_TELEFONE',
                    from_='SEU_NUMERO_TWILIO',
                    body='Detrito detectado com borda visível!'
                )
                print('Mensagem enviada:', message.sid)

    # exiba o frame resultante
    cv2.imshow('Detecção de Objetos', frame)
    key = cv2.waitKey(1) & 0xFF

    # saia do loop se a tecla 'q' for pressionada
    if key == ord('q'):
        break

# limpeza
cap.release()
cv2.destroyAllWindows()
