import cv2
import torch

# Cargar el modelo preentrenado de YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5s' o un modelo personalizado

# Leer el video de la cámara
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detección de manos
    results = model(frame)
    results.render()  # Dibuja las detecciones en el frame

    # Muestra el frame con detecciones
    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
