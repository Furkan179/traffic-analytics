import cv2

def click_event(event, x, y, flags, params):
    # Sol tıka basıldığında koordinatı al ve ekrana nokta çiz
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"[{x}, {y}],")
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Kalibrasyon", img)

cap = cv2.VideoCapture("traffic.mp4")
ret, frame = cap.read()
img = cv2.resize(frame, (640, 480))

cv2.imshow("Kalibrasyon", img)
cv2.setMouseCallback("Kalibrasyon", click_event)

print("Sırasıyla tıkla: Sol Üst -> Sağ Üst -> Sağ Alt -> Sol Alt")
print("Çıkmak için herhangi bir tuşa bas.")

cv2.waitKey(0)
cv2.destroyAllWindows()