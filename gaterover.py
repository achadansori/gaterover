import cv2
import numpy as np
import time

def detect_objects():
    net = cv2.dnn.readNet("yolov4-tiny-custom_gaterover.weights", "yolov4-tiny-custom.cfg")
    output_layers = net.getUnconnectedOutLayersNames()
    classes = []
    with open("obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    cap = cv2.VideoCapture(2)

    start_time = time.time()
    frame_count = 0

    gate_kanan_centers = []
    gate_kiri_centers = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        height, width, channels = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (218, 218), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        confidences = []
        boxes = []
        class_ids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

        gate_kanan_centers = []
        gate_kiri_centers = []

        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                confidence = confidences[i]
                class_id = class_ids[i]
                label = f"{classes[class_id]}: {confidence:.2f}"
                
                x, y, w, h = box
                
                # Calculate the center of the bounding box
                center_x = x + w // 2
                center_y = y + h // 2

                if classes[class_id] == 'gate kanan':
                    gate_kanan_centers.append((center_x, center_y))
                elif classes[class_id] == 'gate kiri':
                    gate_kiri_centers.append((center_x, center_y))

                # Draw a point at the center
                cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw lines between centers of "gate kanan" and "gate kiri"
        for center_kanan in gate_kanan_centers:
            for center_kiri in gate_kiri_centers:
                mid_point = ((center_kanan[0] + center_kiri[0]) // 2, (center_kanan[1] + center_kiri[1]) // 2)
                # Draw a point at the midpoint
                cv2.circle(frame, mid_point, 5, (0, 255, 255), -1)
                # Draw a line connecting centers to the midpoint
                cv2.line(frame, center_kanan, center_kiri, (0, 255, 255), 2)

                # Draw a line connecting the midpoint to the line representing the center of the camera frame
                cv2.line(frame, mid_point, (width // 2, mid_point[1]), (255, 255, 255), 2)

        # Draw a line representing the center of the camera frame
        cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 255, 255), 2)

        # Calculate and display FPS
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = frame_count / elapsed_time

        # Display FPS on the screen
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("YOLO Object Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_objects()
