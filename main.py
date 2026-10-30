import cv2
import numpy as np
import time
import winsound
from openvino import Core

# Function to draw retailer alert and discount message within section boundaries
def draw_discount_text(frame, section_rects, discount_active):
    for i, (x1, x2) in enumerate(section_rects):
        if discount_active[i]:
            text = "20% DISCOUNT \nON THIS SECTION\nFOR NEXT 20 MINUTES"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.72
            thickness = 2
            text_size = cv2.getTextSize(text.split("\n")[0], font, font_scale, thickness)[0]
            line_height = text_size[1] + 6

            # Position the alert inside the section
            text_x = x1 + max(10, (x2 - x1) // 8)
            text_y = 50

            # Add a highlighted alert box for the retailer
            box_x1 = max(0, text_x - 15)
            box_y1 = max(0, text_y - 20)
            box_x2 = min(frame.shape[1], text_x + 180)
            box_y2 = min(frame.shape[0], text_y + (3 * line_height) + 20)
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 255), 2)
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 255), -1, lineType=cv2.LINE_AA)

            # Draw text in a readable color on the alert box
            for j, line in enumerate(text.split("\n")):
                cv2.putText(frame, line, (text_x, text_y + j * line_height), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            # Additional retailer-facing summary just below the section
            summary = f"LOW FOOTFALL - SECTION {i + 1}"
            cv2.putText(frame, summary, (x1 + 10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

# Load OpenVINO model
def load_openvino_model(model_path):
    ie = Core()
    model = ie.read_model(model_path)
    compiled_model = ie.compile_model(model, "CPU")
    return compiled_model, compiled_model.inputs[0].shape

# Run inference with OpenVINO
def infer_openvino(model, frame):
    # Convert to RGB (if the model expects RGB input)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize to match the model's expected input size (640x640 for YOLOv8 models)
    input_tensor = cv2.resize(frame_rgb, (640, 640))  # Resize to 640x640
    input_tensor = input_tensor.transpose(2, 0, 1)  # Convert HWC to CHW
    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32) / 255.0  # Normalize
    
    # Get output from model
    output = model([input_tensor])[model.outputs[0]]
    
    # Flip frame horizontally if necessary (for visualization)
    frame = cv2.flip(frame, 1)  # Flip horizontally (if needed)
    
    return output, frame

# Main function
def main(video_path, num_sections):
    model, input_shape = load_openvino_model("yolov8n.xml")  # Path to OpenVINO model
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    section_width = frame_width // num_sections
    section_rects = [(i * section_width, (i + 1) * section_width) for i in range(num_sections)]

    discount_active = [False] * num_sections
    section_timers = [None] * num_sections  # Store start time for discount
    crowded_alert_active = [False] * num_sections
    alert_cooldowns = [0.0] * num_sections

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Perform inference and get output
        detections, frame = infer_openvino(model, frame)
        section_counts = [0] * num_sections  # Reset counts each frame

        for detection in detections[0]:  # Iterate over detections
            x1, y1, x2, y2, conf, cls = detection[:6]

            # Adjust the coordinates if necessary (if x1 > x2, swap them)
            if x1 > x2:
                x1, x2 = x2, x1  # Swap x1 and x2 if they are inverted
            if y1 > y2:
                y1, y2 = y2, y1  # Swap y1 and y2 if they are inverted

            if int(cls) == 0:  # Check if the class is 'person' (class 0)
                center_x = (x1 + x2) / 2
                section_index = int(center_x // section_width)
                section_index = min(section_index, num_sections - 1)
                section_counts[section_index] += 1

        # Update discount status
        for i in range(num_sections):
            if section_counts[i] <= 1:
                if section_timers[i] is None:
                    section_timers[i] = time.time()  # Start timer
                elif time.time() - section_timers[i] > 10:  # Check 10s condition
                    discount_active[i] = True
            else:
                discount_active[i] = False
                section_timers[i] = None  # Reset timer when >2 people are present
            if section_counts[i] > 10:
                discount_active[i] = False  # Remove discount when more than 10 people come

            # Manager alert for crowded section when people > 8
            if section_counts[i] > 8:
                crowded_alert_active[i] = True
                if time.time() - alert_cooldowns[i] > 10:
                    winsound.Beep(1200, 250)
                    alert_cooldowns[i] = time.time()
                    print(f"Manager alert: Section {i + 1} is crowded ({section_counts[i]} people)")
            else:
                crowded_alert_active[i] = False

        draw_discount_text(frame, section_rects, discount_active)  # Display discount

        # Draw crowded-section manager alert text
        for i, (x1, x2) in enumerate(section_rects):
            if crowded_alert_active[i]:
                cv2.putText(frame, f"MANAGER ALERT: SECTION {i + 1} CROWDED", (x1 + 10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

        # Draw section boundaries
        for x1, x2 in section_rects:
            cv2.line(frame, (x1, 0), (x1, frame_height), (255, 255, 255), 2)

        cv2.imshow("Discount Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run with webcam by default, or with a provided video file if desired
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Retail discount detection using OpenVINO + YOLOv8")
    parser.add_argument("--video", type=str, default="Test_video.mp4", help="Path to a video file to process")
    parser.add_argument("--camera", action="store_true", help="Use the webcam instead of a video file")
    parser.add_argument("--sections", type=int, default=3, help="Number of store sections to divide the frame into")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.camera:
        video_path = 0
    else:
        video_path = args.video

    if args.sections < 1:
        raise ValueError("Number of sections must be at least 1.")

    main(video_path, args.sections)