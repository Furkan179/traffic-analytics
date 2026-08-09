import cv2
import math
import numpy as np
from threading import Thread
from queue import Queue
import time
from ultralytics import YOLO
import os

# ==========================================
# 1. THREADING & QUEUE MIMARISI (PRODUCER-CONSUMER)
# ==========================================
class VideoStream:
    def __init__(self, src="traffic.mp4", queue_size=128):
        self.stream = cv2.VideoCapture(src)
        self.stopped = False
        # RAM'i şişirmemek için maksimum 128 karelik bir tampon bölge (buffer) açıyoruz
        self.Q = Queue(maxsize=queue_size) 

    def start(self):
        # Video okuma işlemini ayrı bir çekirdekte başlat
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                break
            
            # Eğer kuyrukta yer varsa videodan yeni kare oku
            if not self.Q.full():
                ret, frame = self.stream.read()
                if not ret:
                    self.stopped = True
                    break
                self.Q.put(frame)
            else:
                # Kuyruk doluysa Thread'i dinlendir (İşlemciyi %100 kullanmasını engeller)
                time.sleep(0.01) 

    def read(self):
        # Video bittiyse ve kuyrukta işlenecek kare kalmadıysa döngüyü kır
        if self.stopped and self.Q.empty():
            return False, None
        
        # Kuyruktan sıradaki kareyi al (FIFO - İlk giren ilk çıkar)
        return True, self.Q.get()

    def stop(self):
        self.stopped = True
        self.stream.release()

# ==========================================
# 2. KUŞ BAKIŞI (HOMOGRAPHY) KALİBRASYONU
# ==========================================
# Çıkarttığın noktalar (Kameradaki yamuk alan)
pts_src = np.array([[283, 136], [472, 135], [605, 334], [151, 351]], dtype=np.float32)

# Gerçek dünyadaki düz dikdörtgen karşılığı (Sanal kuş bakışı alanımız)
pts_dst = np.array([[0, 0], [200, 0], [200, 400], [0, 400]], dtype=np.float32)

# Dönüşüm Matrisi
M_matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

# ==========================================
# 3. YAPAY ZEKA VE SİSTEM HAFIZASI
# ==========================================
model = YOLO("yolov8n.pt")

vehicle_history = {}  # {track_id: (kus_bakisi_X, kus_bakisi_Y)}
speed_history = {}    # {track_id: [hiz1, hiz2, hiz3, ...]}
ticketed_ids = set()  

fps = 30  
# Kuş bakışı düzleminde artık 1 pikselin metre karşılığı her yerde eşittir
pixel_to_meter = 0.08  
SPEED_LIMIT = 60

# Kuyruk destekli videoyu başlat
vs = VideoStream("traffic.mp4").start()

while True:  
    ret, frame = vs.read()
    if not ret or frame is None:
        break

    resize = cv2.resize(frame, (640, 480))

    # Kalibrasyon alanını ekranda görmek için çokgen çizimi (Sarı renkli yamuk)
    cv2.polylines(resize, [np.int32(pts_src)], isClosed=True, color=(0, 255, 255), thickness=2)

    results = model.track(resize, persist=True, verbose=False)

    for result in results:
        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy()

            for box, conf, cls_id, track_id in zip(boxes, confidences, class_ids, track_ids):
                class_name = model.names[int(cls_id)]

                if class_name in ["car", "truck", "bus", "motorcycle"]:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # 1. Kameradaki Merkez Noktası
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    cv2.circle(resize, (cx, cy), 5, (255, 0, 0), -1)

                    # 2. Koordinatı Kuş Bakışı (BEV) Düzlemine Dönüştür
                    pt = np.array([[[cx, cy]]], dtype=np.float32)
                    warped_pt = cv2.perspectiveTransform(pt, M_matrix)[0][0]
                    bev_x, bev_y = warped_pt[0], warped_pt[1] 

                    if track_id in vehicle_history:
                        prev_bev_x, prev_bev_y = vehicle_history[track_id]
                        
                        # Artık sadece Y ekseninde değil, X-Y düzleminde gerçek vektörel öklid uzaklığını buluyoruz
                        distance_pixels = math.dist((bev_x, bev_y), (prev_bev_x, prev_bev_y))
                        
                        speed_m_per_s = (distance_pixels * pixel_to_meter) * fps
                        speed_km_per_h = speed_m_per_s * 3.6

                        # Sadece poligonun içine giren mantıklı hızları listeye ekliyoruz
                        if 5 < speed_km_per_h < 200:
                            if track_id not in speed_history:
                                speed_history[track_id] = []
                            speed_history[track_id].append(speed_km_per_h)
                            speed_history[track_id] = speed_history[track_id][-5:]
                            
                            average_speed = sum(speed_history[track_id]) / len(speed_history[track_id])

                            if average_speed > SPEED_LIMIT and track_id not in ticketed_ids:
                                ticketed_ids.add(track_id)
                                print(f"Sistem: Hız limiti aşıldı! ID: {track_id}, Hız: {average_speed:.2f} km/h")
                                car_image = resize[y1:y2, x1:x2]
                                if car_image.size > 0:

                                    # 1. Kaydetmek istediğiniz klasör yolunu tanımlayın
                                    klasor_yolu = r"/home/furkan/traffic_analytics_baykar/speed_violation_images"

                                    # 2. Klasör yoksa otomatik oluşturun (Hata almamak için kritik adım)
                                    if not os.path.exists(klasor_yolu):
                                        os.makedirs(klasor_yolu)

                                    # 3. Klasör yolu ile f-string dosya adını birleştirin
                                    tam_yol = os.path.join(klasor_yolu, f"speed_violation_{track_id}_{average_speed:.2f}.jpg")

                                    cv2.imwrite(tam_yol, car_image)
                                cv2.rectangle(resize, (x1, y1), (x2, y2), (0, 0, 255), 2)

                            cv2.putText(resize, f"Hiz: {average_speed:.0f} km/h", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    # Hafızaya kameradaki yanıltıcı 'cy' değerini değil, dönüştürülmüş BEV koordinatlarını kaydediyoruz
                    vehicle_history[track_id] = (bev_x, bev_y)

                    label = f"ID: {int(track_id)} | {class_name}"
                    cv2.rectangle(resize, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(resize, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(resize, f"Ceza Kesilen Arac: {len(ticketed_ids)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)                                  
    cv2.imshow("Trafik Analiz Sistemi", resize)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

vs.stop()
cv2.destroyAllWindows()