from ultralytics import YOLO
import cv2
import mediapipe as mp

# 初始化 YOLO 模型
model = YOLO("yolov8n.pt")  # 輕量版 YOLOv8 模型

# 初始化 MediaPipe 人臉檢測模型
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# 初始化攝像頭
cap = cv2.VideoCapture(0)

while True:
    # 讀取攝像頭影像
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # 使用 YOLO 模型進行全身檢測
    results = model.predict(source=frame, save=False, show=False)

    # 解析 YOLO 檢測結果
    for result in results:
        for box in result.boxes:
            # 提取座標和類別標籤
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 左上角和右下角座標
            confidence = box.conf[0]  # 置信度
            cls = int(box.cls[0])  # 分類 ID
            label = model.names[cls]  # 類別名稱

            # 僅框選 "person" 類別
            if label == "person":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 綠色框
                cv2.putText(
                    frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

    # 使用 MediaPipe 進行臉部檢測
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_detection.process(rgb_frame)

    # 繪製臉部檢測框
    if face_results.detections:
        for detection in face_results.detections:
            # 提取邊界框座標
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                          int(bboxC.width * iw), int(bboxC.height * ih))

            # 繪製框和標籤
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 藍色框
            cv2.putText(
                frame, f"Face {detection.score[0]:.2f}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
            )

    # 顯示結果
    cv2.imshow("YOLOv8 + MediaPipe Face Detection", frame)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
