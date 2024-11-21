import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import time

# Load YOLO model
model = YOLO("yolov8s.pt")


# Mouse callback for debugging (if needed)
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        # print(colorsBGR)


cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

# Video file input
cap = cv2.VideoCapture("parking1.mp4")

# Load class names
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Define parking areas
areas = [
    [(52, 364), (30, 417), (73, 412), (88, 369)],
    [(105, 353), (86, 428), (137, 427), (146, 358)],
    [(159, 354), (150, 427), (204, 425), (203, 353)],
    [(217, 352), (219, 422), (273, 418), (261, 347)],
    [(274, 345), (286, 417), (338, 415), (321, 345)],
    [(336, 343), (357, 410), (409, 408), (382, 340)],
    [(396, 338), (426, 404), (479, 399), (439, 334)],
    [(458, 333), (494, 397), (543, 390), (495, 330)],
    [(511, 327), (557, 388), (603, 383), (549, 324)],
    [(564, 323), (615, 381), (654, 372), (596, 315)],
    [(616, 316), (666, 369), (703, 363), (642, 312)],
    [(674, 311), (730, 360), (764, 355), (707, 308)],
]

# Create or append to the parking space log file
output_file = "parking_space_log.csv"
with open(output_file, "a") as file:
    if file.tell() == 0:  # Add headers if the file is empty
        file.write("Timestamp,AvailableSpaces\n")

# Initialize previous count for logging changes
prev_available_spaces = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    time.sleep(1)
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame, verbose=False)
    px = pd.DataFrame(results[0].boxes.data).astype("float")

    # Lists to track detected cars in each area
    lists = [[] for _ in range(len(areas))]

    for _, row in px.iterrows():
        x1, y1, x2, y2, _, d = row
        d = int(d)
        c = class_list[d]

        if "car" in c:
            cx = int((x1 + x2) // 2)
            cy = int((y1 + y2) // 2)

            for i, area in enumerate(areas):
                result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
                if result >= 0:
                    cv2.rectangle(
                        frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                    )
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                    lists[i].append(c)
                    cv2.putText(
                        frame,
                        c,
                        (int(x1), int(y1)),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

    # Count cars in each area
    counts = [len(lst) for lst in lists]
    total_occupied = sum(counts)
    total_spaces = len(areas) - total_occupied

    # Log data only if the available space count changes
    if prev_available_spaces is None or total_spaces != prev_available_spaces:
        prev_available_spaces = total_spaces
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(output_file, "a") as file:
            file.write(f"{timestamp},{total_spaces}\n")

    # Draw areas and occupancy status
    for i, (area, count) in enumerate(zip(areas, counts)):
        color = (0, 0, 255) if count == 1 else (0, 255, 0)
        cv2.polylines(frame, [np.array(area, np.int32)], True, color, 2)
        cv2.putText(
            frame,
            str(i + 1),
            tuple(np.mean(area, axis=0, dtype=int)),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (255, 255, 255) if count != 1 else (0, 0, 255),
            1,
        )

    # Display total available spaces
    cv2.putText(
        frame,
        f"Spaces: {total_spaces}",
        (23, 30),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        (255, 255, 255),
        2,
    )

    # Display frame
    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
