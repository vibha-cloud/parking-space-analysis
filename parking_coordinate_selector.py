import cv2
import json
import numpy as np

# Initialize list to hold parking space coordinates
parking_coordinates = []


# Mouse callback function to capture coordinates
def select_parking_spot(event, x, y, flags, param):
    global points, image, image_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        # Store clicked points
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Red circle on clicked points
        cv2.imshow("Select Parking Spot", image)

        # If 4 points are selected, draw a green rectangle
        if len(points) == 4:
            # Draw the green rectangle using the four points
            pts = np.array(points, dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            parking_coordinates.append(
                points.copy()
            )  # Save the coordinates for the parking spot
            print(f"Parking space {len(parking_coordinates)} added: {points}")
            points = []  # Reset points for the next parking space
            if len(parking_coordinates) >= 1:
                print("Press 'q' to save and exit.")


# Load parking lot image
image = cv2.imread("parking_lot_image.png")  # Update with your image file
image_copy = image.copy()

# List to store selected points for each parking space
points = []

# Set the mouse callback to select coordinates
cv2.namedWindow("Select Parking Spot")
cv2.setMouseCallback("Select Parking Spot", select_parking_spot)

# Display the image and wait for user input
while True:
    cv2.imshow("Select Parking Spot", image)
    key = cv2.waitKey(1) & 0xFF

    # Exit and save coordinates when 'q' is pressed
    if key == ord("q"):
        with open("parking_coordinates.json", "w") as f:
            json.dump(parking_coordinates, f, indent=4)
        print(f"Parking coordinates saved to parking_coordinates.json")
        break

    # Reset the image if 'r' is pressed (for retry)
    elif key == ord("r"):
        image = image_copy.copy()
        parking_coordinates = []
        print("Resetting coordinates.")

cv2.destroyAllWindows()
