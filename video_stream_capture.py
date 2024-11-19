import cv2

# 定義 RTSP 視訊流 URL
rtsp_url = "rtsp://admin:admin_thinker@192.168.1.136/cam/realmonitor?channel=1&subtype=0" 

# 建立 VideoCapture 物件以從 RTSP URL 獲取視訊流
cap = cv2.VideoCapture(rtsp_url)

# 進入主迴圈以讀取並處理視訊幀
while True:
    # 讀取視訊幀，ret 表示是否成功讀取，frame 為當前幀影像
    ret, frame = cap.read()
    if not ret:  # 若無法讀取視訊幀，結束迴圈
        break

    # 此處可進行影像辨識或處理邏輯（例如物件檢測或影像分析）
    # 顯示當前幀影像於視窗中
    cv2.imshow('frame', frame)

    # 檢查鍵盤輸入，若按下 'q' 鍵則退出迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源，關閉視訊流
cap.release()
# 關閉所有視窗
cv2.destroyAllWindows()
