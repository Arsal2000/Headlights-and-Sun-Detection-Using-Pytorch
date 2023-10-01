import torch
import cv2

model = torch.hub.load('yolov5', 'custom', path='yolov5\\runs\\train\\exp2\\weights\\best.pt', source='local')  # local repo

# Open a video capture object
cap = cv2.VideoCapture(0)  # 0 for default camera, or provide the path to a video file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame
    results = model(frame)

    # Visualize the results on the frame
    frame_with_results = results.render()[0]

    # Display the frame with detection results
    cv2.imshow('Headlight - Sun Real time Detection', frame_with_results)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()