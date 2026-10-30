import csv
import json
import os
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from openvino import Core


DEFAULT_ZONES = ["Entire Frame"]
TRACK_LOST_TIMEOUT_SECONDS = 2.0
INITIAL_OCCUPANCY_WINDOW_SECONDS = 2.0
PERSON_CONFIDENCE_THRESHOLD = 0.30

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
_INFERENCE_LOCK = threading.Lock()

try:
    import winsound
except ImportError:  # pragma: no cover
    winsound = None


@st.cache_resource
def load_model(model_path: str):
    ie = Core()
    model = ie.read_model(model_path)
    compiled_model = ie.compile_model(model, "CPU")
    return compiled_model


def get_default_zone_rects(frame_w: int, frame_h: int, zones):
    zone_count = len(zones)
    rects = []
    if zone_count <= 0:
        return rects

    section_w = frame_w / zone_count
    for index, zone_name in enumerate(zones):
        x1 = int(index * section_w)
        x2 = int((index + 1) * section_w)
        rects.append({
            "name": zone_name,
            "x1": x1,
            "x2": x2,
            "y1": 0,
            "y2": frame_h,
        })
    return rects


def zone_for_point(x: float, y: float, zone_rects):
    for zone in zone_rects:
        if zone["x1"] <= x <= zone["x2"] and zone["y1"] <= y <= zone["y2"]:
            return zone["name"]
    return "Unassigned"


def is_near_frame_edge(x: float, y: float, frame_w: int, frame_h: int):
    margin_x = max(16.0, frame_w * 0.08)
    margin_y = max(16.0, frame_h * 0.08)
    return (
        x <= margin_x
        or x >= frame_w - margin_x
        or y <= margin_y
        or y >= frame_h - margin_y
    )


def infer_model(model, frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_h, frame_w = frame_rgb.shape[:2]
    scale = min(640.0 / frame_w, 640.0 / frame_h)
    resized_w = max(1, int(round(frame_w * scale)))
    resized_h = max(1, int(round(frame_h * scale)))
    resized = cv2.resize(frame_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (640 - resized_w) // 2
    pad_y = (640 - resized_h) // 2
    input_tensor = np.full((640, 640, 3), 114, dtype=np.uint8)
    input_tensor[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized
    input_tensor = input_tensor.transpose(2, 0, 1)
    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32) / 255.0
    with _INFERENCE_LOCK:
        output = model([input_tensor])[model.outputs[0]]
    return output


def get_person_detections(
    model_output,
    frame_w,
    frame_h,
    conf_threshold=PERSON_CONFIDENCE_THRESHOLD,
):
    """Decode OpenVINO YOLOv8 output format [1, 84, 8400].
    Standard COCO YOLOv8 exports have class index 0 = person.
    """
    persons = []
    if model_output is None or len(model_output.shape) != 3:
        return persons

    output = model_output[0]
    if output.shape[0] < 5:
        return persons

    # YOLOv8 output has four box channels followed by 80 class scores.
    boxes = output[:4, :]
    class_scores = output[4:, :]

    if class_scores.size == 0:
        return persons

    class_ids = class_scores.argmax(axis=0)
    confidences = class_scores[class_ids, np.arange(class_scores.shape[1])]

    keep_mask = (class_ids == 0) & (confidences >= conf_threshold)
    indices = np.nonzero(keep_mask)[0]
    if not len(indices):
        return persons

    nms_boxes = []
    nms_scores = []
    candidate_indices = []

    for idx in indices:
        x_center = float(boxes[0, idx])
        y_center = float(boxes[1, idx])
        width = float(boxes[2, idx])
        height = float(boxes[3, idx])

        scale = min(640.0 / frame_w, 640.0 / frame_h)
        resized_w = max(1, int(round(frame_w * scale)))
        resized_h = max(1, int(round(frame_h * scale)))
        scale_x = resized_w / frame_w
        scale_y = resized_h / frame_h
        pad_x = (640 - resized_w) // 2
        pad_y = (640 - resized_h) // 2
        x1 = (x_center - width / 2.0 - pad_x) / scale_x
        y1 = (y_center - height / 2.0 - pad_y) / scale_y
        x2 = (x_center + width / 2.0 - pad_x) / scale_x
        y2 = (y_center + height / 2.0 - pad_y) / scale_y

        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        x1 = max(0.0, min(float(frame_w), x1))
        y1 = max(0.0, min(float(frame_h), y1))
        x2 = max(0.0, min(float(frame_w), x2))
        y2 = max(0.0, min(float(frame_h), y2))
        box_w = int(round(x2 - x1))
        box_h = int(round(y2 - y1))
        if box_w <= 0 or box_h <= 0:
            continue

        nms_boxes.append([int(round(x1)), int(round(y1)), box_w, box_h])
        nms_scores.append(float(confidences[idx]))
        candidate_indices.append((idx, x1, y1, x2, y2))

    if not nms_boxes:
        return persons

    selected = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, conf_threshold, 0.45)
    for selected_index in np.asarray(selected).reshape(-1):
        idx, x1, y1, x2, y2 = candidate_indices[int(selected_index)]
        persons.append({
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "center_x": (float(x1) + float(x2)) / 2.0,
            "center_y": (float(y1) + float(y2)) / 2.0,
            "confidence": float(confidences[idx]),
            "class_id": 0,
        })

    return persons


def box_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def update_tracking(tracked, detections, frame_w, frame_h, zone_rects, frame_index, current_time, next_person_id):
    frame_diagonal = float(np.hypot(frame_w, frame_h))
    candidate_matches = []

    for person_id, info in tracked.items():
        frame_gap = frame_index - info["last_seen_frame"]
        missed_seconds = current_time - info["last_seen_time"]
        if frame_gap <= 0 or missed_seconds > TRACK_LOST_TIMEOUT_SECONDS:
            continue
        velocity_x, velocity_y = info.get("velocity", (0.0, 0.0))
        last_x, last_y = info["last_center"]
        predicted_x = last_x + velocity_x * frame_gap
        predicted_y = last_y + velocity_y * frame_gap
        last_box = info["bbox"]
        predicted_box = (
            last_box[0] + velocity_x * frame_gap,
            last_box[1] + velocity_y * frame_gap,
            last_box[2] + velocity_x * frame_gap,
            last_box[3] + velocity_y * frame_gap,
        )
        box_height = max(1.0, last_box[3] - last_box[1])
        distance_limit = min(
            max(50.0, box_height * 0.9 + np.hypot(velocity_x, velocity_y) * frame_gap * 1.5),
            max(50.0, frame_diagonal * 0.18),
        )

        for detection_index, detection in enumerate(detections):
            dx = detection["center_x"] - predicted_x
            dy = detection["center_y"] - predicted_y
            distance = float(np.hypot(dx, dy))
            detected_box = (detection["x1"], detection["y1"], detection["x2"], detection["y2"])
            overlap = box_iou(predicted_box, detected_box)
            if distance > distance_limit and overlap < 0.05:
                continue
            normalized_distance = min(distance / distance_limit, 1.0)
            cost = 0.75 * normalized_distance + 0.25 * (1.0 - overlap)
            if cost <= 0.9:
                candidate_matches.append((cost, person_id, detection_index))

    candidate_matches.sort()
    matched_ids = set()
    matched_detection_indices = set()
    for _, person_id, detection_index in candidate_matches:
        if person_id in matched_ids or detection_index in matched_detection_indices:
            continue
        info = tracked[person_id]
        detection = detections[detection_index]
        old_x, old_y = info["last_center"]
        frame_gap = frame_index - info["last_seen_frame"]
        observed_velocity = (
            (detection["center_x"] - old_x) / frame_gap,
            (detection["center_y"] - old_y) / frame_gap,
        )
        old_velocity = info.get("velocity", (0.0, 0.0))
        info["velocity"] = (
            0.5 * old_velocity[0] + 0.5 * observed_velocity[0],
            0.5 * old_velocity[1] + 0.5 * observed_velocity[1],
        )
        info["last_center"] = (detection["center_x"], detection["center_y"])
        info["path"].append(info["last_center"])
        if len(info["path"]) > 600:
            info["path"].pop(0)
        info["last_seen_frame"] = frame_index
        info["last_seen_time"] = current_time
        info["inactive_frames"] = 0
        info["visible"] = True
        info["hits"] += 1
        info["entry_event"] = False
        info["exit_event"] = False
        info["confirmation_event"] = False
        old_near_edge = is_near_frame_edge(old_x, old_y, frame_w, frame_h)
        new_near_edge = is_near_frame_edge(
            detection["center_x"],
            detection["center_y"],
            frame_w,
            frame_h,
        )
        if old_near_edge and not new_near_edge:
            info["entry_event"] = True
            info["entry_time"] = current_time
        elif not old_near_edge and new_near_edge:
            info["exit_event"] = True
            info["exit_time"] = current_time
        if not info["confirmed"] and info["hits"] >= 2:
            info["confirmed"] = True
            info["confirmation_event"] = True
        info["bbox"] = (detection["x1"], detection["y1"], detection["x2"], detection["y2"])
        matched_ids.add(person_id)
        matched_detection_indices.add(detection_index)

    for detection_index, detection in enumerate(detections):
        if detection_index in matched_detection_indices:
            continue
        center = (detection["center_x"], detection["center_y"])
        new_id = next_person_id
        next_person_id += 1
        first_zone = zone_for_point(*center, zone_rects)
        tracked[new_id] = {
            "last_center": center,
            "initial_center": center,
            "velocity": (0.0, 0.0),
            "path": [center],
            "hits": 1,
            "confirmed": False,
            "visible": True,
            "entry_event": False,
            "exit_event": False,
            "confirmation_event": False,
            "entry_time": None,
            "exit_time": None,
            "first_seen": current_time,
            "last_seen_time": current_time,
            "last_seen_frame": frame_index,
            "inactive_frames": 0,
            "zone": first_zone,
            "zone_entry_time": current_time,
            "zone_history": [(first_zone, current_time)],
            "dwell_by_zone": defaultdict(float),
            "position": center,
            "bbox": (detection["x1"], detection["y1"], detection["x2"], detection["y2"]),
            "entry_log": [(first_zone, current_time)],
            "exit_log": [],
        }
        matched_ids.add(new_id)

    for person_id in list(tracked):
        if person_id not in matched_ids:
            tracked[person_id]["visible"] = False
            tracked[person_id]["entry_event"] = False
            tracked[person_id]["exit_event"] = False
            tracked[person_id]["confirmation_event"] = False
            tracked[person_id]["inactive_frames"] = frame_index - tracked[person_id]["last_seen_frame"]
            if current_time - tracked[person_id]["last_seen_time"] > TRACK_LOST_TIMEOUT_SECONDS:
                del tracked[person_id]

    for info in tracked.values():
        current_zone = zone_for_point(info["last_center"][0], info["last_center"][1], zone_rects)
        if current_zone != info["zone"]:
            previous_zone = info["zone"]
            dwell = current_time - info["zone_entry_time"]
            if previous_zone not in ["Unassigned", None]:
                info["dwell_by_zone"][previous_zone] += dwell
                info["exit_log"].append((previous_zone, current_time, dwell))
            info["entry_log"].append((current_zone, current_time))
            info["zone_history"].append((current_zone, current_time))
            info["zone"] = current_zone
            info["zone_entry_time"] = current_time
        info["position"] = info["last_center"]

    return tracked, next_person_id


def count_tracked_by_zone(tracked, zone_rects):
    counts = {zone["name"]: 0 for zone in zone_rects}
    for info in tracked.values():
        if not info.get("confirmed", True) or not info.get("visible", True):
            continue
        zone_name = zone_for_point(info["last_center"][0], info["last_center"][1], zone_rects)
        if zone_name in counts:
            counts[zone_name] += 1
    return counts


def update_congestion_state(zone_counts, zone_rects, threshold, active_zones, alerts, sound_enabled):
    for zone in zone_rects:
        zone_name = zone["name"]
        count = zone_counts.get(zone_name, 0)
        state = active_zones.get(zone_name)
        started = state is None
        reset_threshold = max(0, threshold - 1)
        if count >= threshold or (state is not None and count >= reset_threshold):
            if started:
                state = {"count": count}
                active_zones[zone_name] = state
            else:
                state["count"] = count
            level = "HIGH" if count >= threshold else "ELEVATED"
            if zone_name == "Entire Frame":
                state["message"] = f"{level} CUSTOMER VOLUME - {count} customers in the frame"
            else:
                state["message"] = f"{level} CUSTOMER DENSITY - {zone_name}: {count} customers"
            if started and sound_enabled and winsound is not None:
                try:
                    winsound.Beep(1200, 250)
                except (RuntimeError, OSError) as exc:
                    print(f"Sound alert could not be played: {exc}")
        elif state is not None:
            del active_zones[zone_name]

    alerts[:] = [
        active_zones[zone["name"]]["message"]
        for zone in zone_rects
        if zone["name"] in active_zones
    ]


def draw_section_overlays(frame, zone_rects, section_counts, congestion_threshold, low_threshold):
    frame_h, frame_w = frame.shape[:2]
    for zone in zone_rects:
        x1, x2 = zone["x1"], zone["x2"]
        cv2.rectangle(frame, (int(x1), 0), (int(x2), frame_h), (255, 255, 255), 2)
        cv2.putText(
            frame,
            zone["name"],
            (int(x1 + 10), 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        count = section_counts.get(zone["name"], 0)
        label = f"{zone['name']}: {count}"
        cv2.putText(
            frame,
            label,
            (int(x1 + 10), 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if count >= congestion_threshold:
            cv2.putText(
                frame,
                "HIGH CUSTOMER DENSITY",
                (int(x1 + 10), 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"{zone['name']}: {count} customers",
                (int(x1 + 10), 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        elif count <= low_threshold:
            cv2.putText(
                frame,
                "Low traffic",
                (int(x1 + 10), 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )


def build_heatmap(zone_rects, tracked, frame_w, frame_h):
    heatmap_h, heatmap_w = 600, 900
    heatmap = np.zeros((heatmap_h, heatmap_w), dtype=np.float32)
    for info in tracked.values():
        if not info["path"]:
            continue
        for px, py in info["path"]:
            x = int(px * heatmap_w / max(frame_w, 1))
            y = int(py * heatmap_h / max(frame_h, 1))
            if 0 <= x < heatmap.shape[1] and 0 <= y < heatmap.shape[0]:
                heatmap[y, x] += 1.0

    if heatmap.max() > 0:
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=12)
        heatmap = heatmap / heatmap.max()
    return heatmap


def build_heatmap_from_grid(traffic_grid):
    grid = np.asarray(traffic_grid, dtype=np.float32)
    if grid.ndim != 2 or grid.size == 0:
        return np.zeros((600, 900), dtype=np.float32)
    grid = np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)
    if float(grid.max()) <= 0:
        return np.zeros((600, 900), dtype=np.float32)
    heatmap = cv2.GaussianBlur(grid, (0, 0), sigmaX=1.5)
    heatmap = cv2.resize(heatmap, (900, 600), interpolation=cv2.INTER_LINEAR)
    maximum = float(heatmap.max())
    return heatmap / maximum if maximum > 0 else np.zeros_like(heatmap)


def accumulate_heatmap_positions(traffic_grid, tracked, frame_w, frame_h):
    grid_height, grid_width = traffic_grid.shape
    samples = 0
    for info in tracked.values():
        if not info.get("confirmed", False) or not info.get("visible", False):
            continue
        center_x, center_y = info["last_center"]
        if not np.isfinite(center_x) or not np.isfinite(center_y):
            continue
        grid_x = min(grid_width - 1, max(0, int(center_x * grid_width / max(frame_w, 1))))
        grid_y = min(grid_height - 1, max(0, int(center_y * grid_height / max(frame_h, 1))))
        traffic_grid[grid_y, grid_x] += 1.0
        samples += 1
    return samples


def open_ip_stream(stream_url: str):
    """Open an RTSP/HTTP stream without exposing the URL (which may contain credentials)."""
    from urllib.parse import urlsplit

    try:
        parsed_url = urlsplit(stream_url.strip())
        if parsed_url.scheme.lower() not in {"rtsp", "rtsps", "http", "https"} or not parsed_url.hostname:
            raise ValueError
    except ValueError as exc:
        raise ValueError("Enter a valid RTSP, RTSPS, HTTP, or HTTPS stream URL.") from exc

    try:
        capture = cv2.VideoCapture(
            stream_url.strip(),
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                5000,
            ],
        )
    except cv2.error as exc:
        raise ValueError("OpenCV could not initialize this stream. Check the URL and supported stream format.") from exc

    if not capture.isOpened():
        capture.release()
        raise ValueError("Could not connect to the stream. Check the URL, credentials, network, and camera availability.")

    ok, first_frame = capture.read()
    if not ok or first_frame is None:
        capture.release()
        raise ValueError("Connected to the stream, but no video frame was received. Check the stream path and camera.")
    return capture, first_frame


def describe_hotspot(heatmap):
    if heatmap is None or heatmap.size == 0 or float(heatmap.max()) <= 0:
        return "Collecting movement data"
    y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    return f"Image area near {x * 100 // heatmap.shape[1]}% across, {y * 100 // heatmap.shape[0]}% down"


def describe_movement(tracked):
    directions = []
    for info in tracked.values():
        path = info.get("path", [])
        if len(path) < 2:
            continue
        dx = path[-1][0] - path[0][0]
        dy = path[-1][1] - path[0][1]
        if abs(dx) + abs(dy) < 20:
            continue
        horizontal = "right" if dx > 0 else "left" if dx < 0 else ""
        vertical = "down" if dy > 0 else "up" if dy < 0 else ""
        directions.append("-".join(part for part in (vertical, horizontal) if part))
    if not directions:
        return "Movement direction is still being established"
    return f"Common direction: {pd.Series(directions).mode().iloc[0]}"


def describe_traffic_periods(activity_buckets, duration_seconds=None):
    if not activity_buckets:
        return "Collecting traffic history", "Collecting traffic history"
    if duration_seconds is not None and duration_seconds < 120:
        return "Insufficient duration for period comparison", "Insufficient duration for period comparison"
    busiest = max(
        activity_buckets,
        key=lambda minute: activity_buckets[minute]["active_total"]
        / max(1, activity_buckets[minute]["samples"]),
    )
    quietest = min(
        activity_buckets,
        key=lambda minute: activity_buckets[minute]["active_total"]
        / max(1, activity_buckets[minute]["samples"]),
    )
    return f"Minute {busiest + 1} of this session", f"Minute {quietest + 1} of this session"


def format_elapsed_time(seconds):
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours} hr {minutes} min {seconds} sec"
    return f"{minutes} min {seconds} sec"


def classify_crowd(count, medium_threshold, high_threshold):
    if count >= high_threshold:
        return "HIGH", 3
    if count >= medium_threshold:
        return "MEDIUM", 2
    return "LOW", 1


def summarize_customer_records(
    customer_records,
    activity_buckets,
    crowd_levels,
    current_customers=0,
    active_tracked_customers=0,
    peak_activity_time_seconds=0.0,
    duration_seconds=None,
):
    dwell_values = [
        max(
            0.0,
            row.get(
                "Observed Track Duration (seconds)",
                row.get("Dwell Time (seconds)", row.get("dwell_seconds", 0.0)),
            ),
        )
        for row in customer_records
    ]
    entries = sum(int(row.get("Entry Count", 0)) for row in customer_records)
    exits = sum(int(row.get("Exit Count", 0)) for row in customer_records)
    initial_occupancy = sum(
        row.get("Occupancy Origin") == "initial" for row in customer_records
    )
    occupancy_estimate = initial_occupancy + entries - exits
    occupancy_unknown = occupancy_estimate < 0 or any(
        row.get("Status", row.get("status")) in {"lost", "unclassified"}
        or row.get("Occupancy Origin") == "unclassified"
        for row in customer_records
    )
    average_dwell = float(np.mean(dwell_values)) if dwell_values else 0.0
    maximum_dwell = max(dwell_values, default=0.0)

    peak_minute = None
    peak_count = 0.0
    if activity_buckets:
        peak_minute = max(
            activity_buckets,
            key=lambda minute: activity_buckets[minute]["active_total"]
            / max(1, activity_buckets[minute]["samples"]),
        )
        peak_count = (
            activity_buckets[peak_minute]["active_total"]
            / max(1, activity_buckets[peak_minute]["samples"])
        )
    if peak_minute is None:
        peak_period = "Not available (no activity samples)"
    elif duration_seconds is not None and duration_seconds < 120:
        peak_period = "Insufficient clip duration (<2 min)"
    else:
        peak_period = (
            f"{format_elapsed_time(peak_minute * 60)} - "
            f"{format_elapsed_time((peak_minute + 1) * 60)}"
        )
    average_crowd = float(np.mean(crowd_levels)) if crowd_levels else 0.0
    return {
        "total_unique_customers": len(customer_records),
        "current_customers": current_customers,
        "active_tracked_customers": active_tracked_customers,
        "currently_inside": None if occupancy_unknown else occupancy_estimate,
        "occupancy_estimate": max(0, occupancy_estimate),
        "occupancy_status": "unknown" if occupancy_unknown else "estimated",
        "initial_occupancy_estimate": initial_occupancy,
        "total_entries": entries,
        "total_exits": exits,
        "average_dwell_seconds": average_dwell,
        "maximum_dwell_seconds": maximum_dwell,
        "peak_period": peak_period,
        "peak_activity_time_seconds": peak_activity_time_seconds,
        "maximum_crowd": max(
            (bucket["maximum"] for bucket in activity_buckets.values()),
            default=0,
        ),
        "average_crowd_level": (
            "LOW" if average_crowd < 1.5
            else "MEDIUM" if average_crowd < 2.5
            else "HIGH"
        ),
        "peak_minute_average": peak_count,
    }


def add_session_metrics_to_records(customer_records, analytics):
    metrics = {
        "Session Current Visible Customers": analytics["current_customers"],
        "Session Active Tracked Customers": analytics["active_tracked_customers"],
        "Session Total Unique Tracked IDs": analytics["total_unique_customers"],
        "Session Estimated Occupancy": (
            analytics["currently_inside"]
            if analytics["currently_inside"] is not None
            else "Unknown"
        ),
        "Session Occupancy Status": analytics["occupancy_status"],
        "Session Initial Occupancy Estimate": analytics["initial_occupancy_estimate"],
        "Session Entries": analytics["total_entries"],
        "Session Exits": analytics["total_exits"],
        "Session Peak Visible Customers": analytics["maximum_crowd"],
        "Session Average Observed Duration (seconds)": analytics["average_dwell_seconds"],
        "Session Maximum Observed Duration (seconds)": analytics["maximum_dwell_seconds"],
        "Session Peak Activity Time (seconds)": analytics["peak_activity_time_seconds"],
        "Session Peak Period": analytics["peak_period"],
        "Session Heatmap Position Samples": analytics.get("heatmap_position_samples", 0),
        "Session Active Congestion Alerts": "; ".join(analytics.get("congestion_alerts", [])),
    }
    return [{**record, **metrics} for record in customer_records]


def traffic_frame(activity_buckets):
    rows = []
    for minute, bucket in sorted(activity_buckets.items()):
        samples = max(1, bucket["samples"])
        rows.append({
            "Minute": f"{minute:02d}:00",
            "Entries per minute": bucket["unique_entries"],
            "Active customers": bucket["active_total"] / samples,
        })
    return pd.DataFrame(rows)


def save_analysis_records(records, file_base):
    csv_path = file_base.with_suffix(".csv")
    json_path = file_base.with_suffix(".json")

    pd.DataFrame(records).to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(records, fp, indent=2)

    return csv_path, json_path


def create_analysis_report(summary):
    most_visited = summary["most_visited_section"]
    least_visited = summary["least_visited_section"]
    avg_dwell = summary["average_dwell_time_seconds"]
    avg_dwell_text = (
        f"{avg_dwell:.1f} seconds"
        if avg_dwell is not None
        else "Not available for this video."
    )
    occupancy = summary.get("currently_inside")
    occupancy_text = str(occupancy) if occupancy is not None else "Unknown"
    report = f"""
AI Retail Customer Experience Analysis Report
=========================================

Total Unique Tracked IDs Across Video: {summary['total_detected_customers']}
Maximum person detections in an analyzed frame: {summary['peak_customers']}
Person detections on final analyzed frame: {summary['current_customers']}
Active tracked customers at end: {summary.get('active_tracked_customers', 0)}
Confirmed tracked position samples for heatmap: {summary.get('heatmap_position_samples', 0)}
Total entries: {summary.get('entry_count', 0)}
Total exits: {summary.get('exit_count', 0)}
Estimated occupancy: {occupancy_text}
Occupancy confidence: {summary.get('occupancy_status', 'unknown')}
Initial occupancy estimate: {summary.get('initial_occupancy_estimate', 0)}
Most visited area: {most_visited}
Least visited area: {least_visited}
Average observed track duration: {avg_dwell_text}
Maximum observed track duration: {format_elapsed_time(summary.get('maximum_dwell_time_seconds', 0))}
Peak period within clip: {summary.get('peak_period', 'Not available')}
Peak activity: {summary.get('peak_activity_time_seconds', 0):.1f} seconds from clip start
Busy period: {summary.get('busiest_period', 'Not available')}
Lower-traffic period: {summary.get('quietest_period', 'Not available')}
Frequently visited area: {summary.get('most_visited_area', 'Not available')}
Movement pattern: {summary.get('movement_pattern', 'Not available')}

High-volume alerts:
{summary['congestion_alerts']}

Checkout/queue insight:
{summary['checkout_status']}

Important observations:
{summary['notes']}

Metric definitions:
Visible detections are person boxes on the most recently analyzed frame; brief unconfirmed detections may be included. Active tracks are confirmed IDs retained for up to {TRACK_LOST_TIMEOUT_SECONDS:.1f} seconds after their last detection. Unique tracked IDs count confirmed anonymous tracks. Initial occupancy is estimated from confirmed tracks first seen in the initial {INITIAL_OCCUPANCY_WINDOW_SECONDS:.1f} seconds. Boundary entries/exits are confirmed crossings between the edge band (8% of frame dimension, minimum 16 pixels) and the interior, in the corresponding inward/outward direction. Occupancy is initial estimate + valid entries - valid exits; it is labeled unknown if unclassified or lost tracks make the ledger incomplete. Observed track duration is first-seen to last-seen time, including gaps bridged by the tracker; it is not verified time inside the store. Peak times are relative to the clip start and are not business-hour estimates.
"""
    return report


def process_video(
    video_file,
    zone_names,
    congestion_threshold=8,
    low_threshold=1,
    sound_enabled=True,
    medium_crowd_threshold=5,
    high_crowd_threshold=12,
    progress_callback=None,
):
    model = load_model(str(PROJECT_DIR / "yolov8n.xml"))
    video_path = Path(video_file)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    zone_rects = get_default_zone_rects(frame_width, frame_height, zone_names)
    tracked = {}
    next_person_id = 1
    customer_records = {}
    records = []
    alerts = []
    peak_customers = 0
    peak_time = 0.0
    current_customers = 0
    frame_index = 0
    current_time = 0.0
    final_zone_counts = defaultdict(int)
    active_congested_zones = {}
    traffic_grid = np.zeros((30, 40), dtype=np.float32)
    heatmap_position_samples = 0
    activity_buckets = {}
    crowd_levels = []
    inference_stride = 2

    output_video_name = f"processed_{uuid.uuid4().hex}.mp4"
    output_video_path = OUTPUT_DIR / output_video_name
    preview_scale = min(1.0, 1280.0 / max(frame_width, frame_height))
    output_width = max(2, int(frame_width * preview_scale) // 2 * 2)
    output_height = max(2, int(frame_height * preview_scale) // 2 * 2)
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"avc1"),
        int(fps),
        (output_width, output_height),
    )
    if not writer.isOpened():
        writer.release()
        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            int(fps),
            (output_width, output_height),
        )
    if not writer.isOpened():
        cap.release()
        writer.release()
        raise ValueError("Could not create the processed-video output with the available MP4 codecs.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        current_time = frame_index / max(fps, 1.0)
        inference_due = (
            frame_index % inference_stride == 0
            or (total_frames > 0 and frame_index == total_frames - 1)
        )
        if inference_due:
            try:
                detections = get_person_detections(
                    infer_model(model, frame),
                    frame_width,
                    frame_height,
                )
            except Exception:
                cap.release()
                writer.release()
                raise

            tracking_before_update = {
                person_id: {
                    "confirmed": info["confirmed"],
                    "last_seen_time": info["last_seen_time"],
                    "last_center": info["last_center"],
                }
                for person_id, info in tracked.items()
            }
            tracked, next_person_id = update_tracking(
                tracked,
                detections,
                frame_width,
                frame_height,
                zone_rects,
                frame_index,
                current_time,
                next_person_id,
            )
            newly_confirmed_ids = set()
            for person_id, info in tracked.items():
                if info["confirmation_event"] and person_id not in customer_records:
                    occupancy_origin = (
                        "entry" if info["entry_event"]
                        else "initial" if info["first_seen"] <= INITIAL_OCCUPANCY_WINDOW_SECONDS
                        else "unclassified"
                    )
                    customer_records[person_id] = {
                        "Customer ID": person_id,
                        "Entry Time (seconds)": info["entry_time"] or "",
                        "First Seen Time (seconds)": info["first_seen"],
                        "Exit Time (seconds)": info["exit_time"] or "",
                        "Observed Track Duration (seconds)": max(0.0, info["last_seen_time"] - info["first_seen"]),
                        "Entry Count": int(info["entry_event"]),
                        "Exit Count": int(info["exit_event"]),
                        "Occupancy Origin": occupancy_origin,
                        "Status": "exited" if info["exit_event"] else (
                            "inside" if occupancy_origin != "unclassified" else "unclassified"
                        ),
                    }
                    newly_confirmed_ids.add(person_id)
            for person_id, info in tracked.items():
                record = customer_records.get(person_id)
                if record is None or person_id in newly_confirmed_ids:
                    continue
                if info["entry_event"]:
                    record["Entry Count"] += 1
                    if record["Occupancy Origin"] == "unclassified":
                        record["Occupancy Origin"] = "entry"
                    if not record["Entry Time (seconds)"]:
                        record["Entry Time (seconds)"] = current_time
                    record["Status"] = "inside"
                if info["exit_event"]:
                    record["Exit Count"] += 1
                    record["Exit Time (seconds)"] = current_time
                    record["Status"] = "exited"
            for person_id in tracking_before_update.keys() - tracked.keys():
                previous = tracking_before_update[person_id]
                if not previous["confirmed"] or person_id not in customer_records:
                    continue
                record = customer_records[person_id]
                record["Observed Track Duration (seconds)"] = max(
                    0.0,
                    previous["last_seen_time"] - record["First Seen Time (seconds)"],
                )
                if record["Status"] != "exited":
                    record["Status"] = "lost"

            current_customers = len(detections)
            if current_customers > peak_customers:
                peak_customers = current_customers
                peak_time = current_time
            current_zone_counts = count_tracked_by_zone(tracked, zone_rects)
            final_zone_counts = current_zone_counts
            update_congestion_state(
                current_zone_counts,
                zone_rects,
                congestion_threshold,
                active_congested_zones,
                alerts,
                sound_enabled,
            )

            activity_minute = int(current_time // 60)
            bucket = activity_buckets.setdefault(
                activity_minute,
                {
                    "active_total": 0,
                    "unique_entries": 0,
                    "samples": 0,
                    "maximum": 0,
                },
            )
            entries_this_sample = sum(
                info["confirmed"] and info["entry_event"]
                for info in tracked.values()
            )
            bucket["active_total"] += current_customers
            bucket["unique_entries"] += entries_this_sample
            bucket["samples"] += 1
            bucket["maximum"] = max(bucket["maximum"], current_customers)
            crowd_name, crowd_score = classify_crowd(
                current_customers,
                medium_crowd_threshold,
                high_crowd_threshold,
            )
            crowd_levels.append(crowd_score)

            heatmap_position_samples += accumulate_heatmap_positions(
                traffic_grid,
                tracked,
                frame_width,
                frame_height,
            )
            for person_id, record in customer_records.items():
                info = tracked.get(person_id)
                if info is not None and info["confirmed"]:
                    record["Observed Track Duration (seconds)"] = max(
                        0.0,
                        info["last_seen_time"] - record["First Seen Time (seconds)"],
                    )
            if progress_callback and (
                frame_index % max(1, inference_stride * 5) == 0
                or (total_frames > 0 and frame_index + 1 == total_frames)
            ):
                progress = frame_index / max(1, total_frames - 1) if total_frames > 0 else 0.0
                progress_callback(
                    progress,
                    f"Analyzed frame {frame_index + 1:,}"
                    + (f" of {total_frames:,}" if total_frames > 0 else ""),
                )

        crowd_name, _ = classify_crowd(
            current_customers,
            medium_crowd_threshold,
            high_crowd_threshold,
        )

        for person_id, info in tracked.items():
            if not info["confirmed"]:
                continue
            x1, y1, x2, y2 = info["bbox"]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {person_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(
            frame,
            f"Current customers: {current_customers} | Crowd: {crowd_name}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if (output_width, output_height) != (frame_width, frame_height):
            frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()
    if progress_callback:
        progress_callback(1.0, f"Completed {frame_index:,} frames.")

    heatmap = build_heatmap_from_grid(traffic_grid)
    heatmap_image = (heatmap * 255).astype(np.uint8)
    heatmap_image = cv2.applyColorMap(heatmap_image, cv2.COLORMAP_JET)
    heatmap_path = OUTPUT_DIR / f"heatmap_{uuid.uuid4().hex}.png"
    cv2.imwrite(str(heatmap_path), heatmap_image)


    records = list(customer_records.values())
    active_tracked_customers = sum(
        info["confirmed"]
        and current_time - info["last_seen_time"] <= TRACK_LOST_TIMEOUT_SECONDS
        for info in tracked.values()
    )
    duration_seconds = (total_frames - 1) / max(fps, 1.0) if total_frames > 1 else 0.0
    analytics = summarize_customer_records(
        records,
        activity_buckets,
        crowd_levels,
        current_customers,
        active_tracked_customers,
        peak_time,
        duration_seconds,
    )
    analytics["congestion_alerts"] = list(alerts)
    analytics["heatmap_position_samples"] = heatmap_position_samples
    total_detected = analytics["total_unique_customers"]
    current_inside = analytics["currently_inside"]
    entered_count = analytics["total_entries"]
    exited_count = analytics["total_exits"]
    average_dwell = analytics["average_dwell_seconds"]
    maximum_dwell = analytics["maximum_dwell_seconds"]

    section_summary = []
    for zone in zone_rects:
        name = zone["name"]
        count = final_zone_counts.get(name, 0)
        avg_dwell = average_dwell if total_detected else None
        section_summary.append({
            "Zone": name,
            "Visible Tracked Customers": count,
            "Traffic Status": (
                "High traffic" if count >= congestion_threshold
                else "Low traffic" if count <= low_threshold
                else "Normal"
            ),
            "Average Observed Duration": f"{avg_dwell:.1f}s" if avg_dwell is not None else "Not available for this video.",
            "Entries": entered_count,
            "Exits": exited_count,
            "Unique Customers": total_detected,
            "Avg Dwell (s)": avg_dwell if avg_dwell is not None else np.nan,
        })

    section_df = pd.DataFrame(section_summary)
    if section_df.empty:
        section_df = pd.DataFrame({"Zone": [], "Visible Tracked Customers": [], "Entered": [], "Left": [], "Unique Customers": []})

    high_traffic = section_df[section_df["Visible Tracked Customers"] >= congestion_threshold]["Zone"].tolist() if not section_df.empty else []
    low_traffic = section_df[section_df["Visible Tracked Customers"] <= low_threshold]["Zone"].tolist() if not section_df.empty else []

    most_visited = section_df.loc[section_df["Visible Tracked Customers"].idxmax(), "Zone"] if not section_df.empty and not section_df["Visible Tracked Customers"].empty else "Not available"
    least_visited = section_df.loc[section_df["Visible Tracked Customers"].idxmin(), "Zone"] if not section_df.empty and not section_df["Visible Tracked Customers"].empty else "Not available"

    busiest_period, quietest_period = describe_traffic_periods(
        activity_buckets,
        duration_seconds,
    )

    report = create_analysis_report({
        "total_detected_customers": total_detected,
        "current_customers": current_customers,
        "peak_customers": analytics["maximum_crowd"],
        "entry_count": entered_count,
        "exit_count": exited_count,
        "currently_inside": current_inside,
        "active_tracked_customers": analytics["active_tracked_customers"],
        "occupancy_status": analytics["occupancy_status"],
        "occupancy_estimate": analytics["occupancy_estimate"],
        "initial_occupancy_estimate": analytics["initial_occupancy_estimate"],
        "most_visited_section": most_visited,
        "least_visited_section": least_visited,
        "average_dwell_time_seconds": average_dwell,
        "maximum_dwell_time_seconds": maximum_dwell,
        "peak_period": analytics["peak_period"],
        "peak_activity_time_seconds": peak_time,
        "heatmap_position_samples": heatmap_position_samples,
        "duration_seconds": duration_seconds,
        "busiest_period": busiest_period,
        "quietest_period": quietest_period,
        "most_visited_area": describe_hotspot(traffic_grid),
        "movement_pattern": describe_movement(tracked),
        "high_traffic_sections": high_traffic,
        "low_traffic_sections": low_traffic,
        "congestion_alerts": alerts,
        "checkout_status": "Not available for this video",
        "notes": "This analysis shows anonymous movement patterns only. Shelf inventory is not estimated unless actual physical stock data is provided.",
    })
    report_path = OUTPUT_DIR / f"report_{uuid.uuid4().hex}.txt"
    report_path.write_text(report, encoding="utf-8")

    records_path = OUTPUT_DIR / f"records_{uuid.uuid4().hex}.csv"
    export_records = add_session_metrics_to_records(records, analytics)
    customer_df = pd.DataFrame(export_records)
    customer_df.to_csv(records_path, index=False)

    assistance_opportunities = []
    if not section_df.empty:
        for _, row in section_df.iterrows():
            if row["Avg Dwell (s)"] > 30:
                assistance_opportunities.append(f"Possible assistance opportunity in {row['Zone']}: observed track duration is above 30 seconds.")
            if row["Visible Tracked Customers"] >= congestion_threshold:
                assistance_opportunities.append(f"Possible assistance opportunity in {row['Zone']}: high customer density detected.")

    summary = {
        "total_detected_customers": total_detected,
        "current_customers": current_customers,
        "peak_customers": analytics["maximum_crowd"],
        "currently_inside": current_inside,
        "active_tracked_customers": analytics["active_tracked_customers"],
        "occupancy_status": analytics["occupancy_status"],
        "occupancy_estimate": analytics["occupancy_estimate"],
        "initial_occupancy_estimate": analytics["initial_occupancy_estimate"],
        "entry_count": entered_count,
        "exit_count": exited_count,
        "traffic_status": section_df.iloc[0]["Traffic Status"] if not section_df.empty else "Not available",
        "section_summary": section_df.to_dict("records"),
        "high_traffic_sections": high_traffic,
        "low_traffic_sections": low_traffic,
        "congestion_alerts": alerts,
        "average_dwell_time_seconds": average_dwell,
        "maximum_dwell_time_seconds": maximum_dwell,
        "peak_period": analytics["peak_period"],
        "peak_activity_time_seconds": peak_time,
        "heatmap_position_samples": heatmap_position_samples,
        "maximum_crowd": analytics["maximum_crowd"],
        "average_crowd_level": analytics["average_crowd_level"],
        "traffic_data": traffic_frame(activity_buckets).to_dict("records"),
        "busiest_period": busiest_period,
        "quietest_period": quietest_period,
        "most_visited_area": describe_hotspot(traffic_grid),
        "movement_pattern": describe_movement(tracked),
        "assistance_opportunities": assistance_opportunities,
        "output_video": str(output_video_path),
        "heatmap_path": str(heatmap_path),
        "csv_path": str(records_path),
        "report_path": str(report_path),
    }

    return {
        "output_video": str(output_video_path),
        "heatmap_path": str(heatmap_path),
        "section_df": section_df,
        "summary": summary,
        "report": report,
        "records": export_records,
        "alerts": alerts,
        "zone_rects": zone_rects,
        "assistance_opportunities": assistance_opportunities,
    }


def release_live_camera():
    capture = st.session_state.get("_live_camera_capture")
    if capture is not None:
        capture.release()
    st.session_state["_live_camera_capture"] = None
    st.session_state["_live_camera_running"] = False
    st.session_state.pop("_live_pending_frame", None)


@st.fragment(run_every=0.1)
def render_live_camera(
    zone_names,
    congestion_threshold,
    low_traffic_threshold,
    sound_enabled,
    medium_crowd_threshold=5,
    high_crowd_threshold=12,
):
    if not st.session_state.get("_live_camera_running", False):
        return

    capture = st.session_state.get("_live_camera_capture")
    if capture is None:
        st.error("The stream is not available. Connect to the stream again.")
        st.session_state["_live_camera_running"] = False
        return

    frame = st.session_state.pop("_live_pending_frame", None)
    if frame is None:
        ok, frame = capture.read()
        if not ok or frame is None:
            st.error("The stream stopped delivering video frames. Check the camera/network, then reconnect.")
            release_live_camera()
            return

    frame_height, frame_width = frame.shape[:2]
    zone_rects = get_default_zone_rects(frame_width, frame_height, zone_names)
    tracked = st.session_state["_live_tracked"]
    frame_index = st.session_state["_live_frame_index"]
    next_person_id = st.session_state["_live_next_person_id"]
    current_time = time.monotonic() - st.session_state["_live_started_at"]
    model = load_model(str(PROJECT_DIR / "yolov8n.xml"))
    detections = get_person_detections(infer_model(model, frame), frame_width, frame_height)
    previous_track_state = {
        person_id: {
            "first_seen": info["first_seen"],
            "last_seen_time": info["last_seen_time"],
            "last_center": info["last_center"],
            "confirmed": info["confirmed"],
        }
        for person_id, info in tracked.items()
    }
    tracked, next_person_id = update_tracking(
        tracked,
        detections,
        frame_width,
        frame_height,
        zone_rects,
        frame_index,
        current_time,
        next_person_id,
    )
    exited_ids = set(previous_track_state) - set(tracked)
    new_customer_ids = [
        person_id
        for person_id, info in tracked.items()
        if info["confirmation_event"] and person_id not in st.session_state["_live_customer_records_by_id"]
    ]
    for person_id in new_customer_ids:
        info = tracked[person_id]
        occupancy_origin = (
            "entry" if info["entry_event"]
            else "initial" if info["first_seen"] <= INITIAL_OCCUPANCY_WINDOW_SECONDS
            else "unclassified"
        )
        customer_record = {
            "Customer ID": person_id,
            "Entry Time (seconds)": "",
            "First Seen Time (seconds)": info["first_seen"],
            "Exit Time (seconds)": "",
            "Observed Track Duration (seconds)": 0.0,
            "Entry Count": 0,
            "Exit Count": 0,
            "Occupancy Origin": occupancy_origin,
            "Status": "inside" if occupancy_origin != "unclassified" else "unclassified",
        }
        st.session_state["_live_customer_records_by_id"][person_id] = customer_record
        st.session_state["_live_customer_records"].append(customer_record)

    for person_id, info in tracked.items():
        customer_record = st.session_state["_live_customer_records_by_id"].get(person_id)
        if customer_record is None:
            continue
        if info["entry_event"]:
            customer_record["Entry Count"] += 1
            if customer_record["Occupancy Origin"] == "unclassified":
                customer_record["Occupancy Origin"] = "entry"
            if not customer_record["Entry Time (seconds)"]:
                customer_record["Entry Time (seconds)"] = current_time
            customer_record["Status"] = "inside"
        if info["exit_event"]:
            customer_record["Exit Count"] += 1
            customer_record["Exit Time (seconds)"] = current_time
            customer_record["Status"] = "exited"

    for person_id in exited_ids:
        previous_track = previous_track_state[person_id]
        if previous_track["confirmed"]:
            customer_record = st.session_state["_live_customer_records_by_id"].get(person_id)
            if customer_record is not None:
                customer_record["Observed Track Duration (seconds)"] = max(
                    0.0,
                    previous_track["last_seen_time"] - customer_record["First Seen Time (seconds)"],
                )
                if customer_record["Status"] != "exited":
                    customer_record["Status"] = "lost"

    for person_id, info in tracked.items():
        customer_record = st.session_state["_live_customer_records_by_id"].get(person_id)
        if customer_record is not None:
            customer_record["Observed Track Duration (seconds)"] = max(
                0.0,
                info["last_seen_time"] - customer_record["First Seen Time (seconds)"],
            )

    zone_counts = count_tracked_by_zone(tracked, zone_rects)
    update_congestion_state(
        zone_counts,
        zone_rects,
        congestion_threshold,
        st.session_state["_live_active_congested_zones"],
        st.session_state["_live_alerts"],
        sound_enabled,
    )

    activity_minute = int(current_time // 60)
    activity_buckets = st.session_state["_live_activity_buckets"]
    current_customers = len(detections)
    bucket = activity_buckets.setdefault(
        activity_minute,
        {
            "active_total": 0,
            "unique_entries": 0,
            "samples": 0,
            "maximum": 0,
        },
    )
    bucket["active_total"] += current_customers
    bucket["unique_entries"] += sum(
        info["confirmed"] and info["entry_event"]
        for info in tracked.values()
    )
    bucket["samples"] += 1
    bucket["maximum"] = max(bucket["maximum"], current_customers)
    _, crowd_score = classify_crowd(
        current_customers,
        medium_crowd_threshold,
        high_crowd_threshold,
    )
    st.session_state["_live_crowd_levels"].append(crowd_score)

    traffic_grid = st.session_state["_live_traffic_grid"]
    st.session_state["_live_heatmap_position_samples"] += accumulate_heatmap_positions(
        traffic_grid,
        tracked,
        frame_width,
        frame_height,
    )

    if current_customers > st.session_state["_live_peak_customers"]:
        st.session_state["_live_peak_customers"] = current_customers
        st.session_state["_live_peak_time"] = current_time

    for person_id, info in tracked.items():
        if not info["confirmed"]:
            continue
        x1, y1, x2, y2 = info["bbox"]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {person_id}",
            (int(x1), max(20, int(y1) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
    cv2.putText(
        frame,
        f"Customers in frame: {current_customers}",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    st.session_state["_live_tracked"] = tracked
    st.session_state["_live_next_person_id"] = next_person_id
    st.session_state["_live_frame_index"] = frame_index + 1
    active_tracked_customers = sum(
        info["confirmed"]
        and current_time - info["last_seen_time"] <= TRACK_LOST_TIMEOUT_SECONDS
        for info in tracked.values()
    )
    live_analytics = summarize_customer_records(
        st.session_state["_live_customer_records"],
        activity_buckets,
        st.session_state["_live_crowd_levels"],
        current_customers,
        active_tracked_customers,
        st.session_state["_live_peak_time"],
        current_time,
    )
    live_analytics["congestion_alerts"] = list(st.session_state["_live_alerts"])
    live_analytics["heatmap_position_samples"] = st.session_state["_live_heatmap_position_samples"]
    average_dwell = live_analytics["average_dwell_seconds"]
    maximum_dwell = live_analytics["maximum_dwell_seconds"]
    busiest_period, quietest_period = describe_traffic_periods(
        activity_buckets,
        current_time,
    )
    movement_heatmap = cv2.GaussianBlur(traffic_grid, (0, 0), sigmaX=1.5)
    movement_heatmap = cv2.resize(movement_heatmap, (320, 200), interpolation=cv2.INTER_LINEAR)
    if movement_heatmap.max() > 0:
        movement_heatmap /= movement_heatmap.max()
    movement_heatmap = cv2.applyColorMap((movement_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

    video_column, insights_column = st.columns([1.4, 1])
    with video_column:
        st.image(frame, channels="BGR", use_container_width=True)
        st.caption("Anonymous people detection and tracking across the full video frame.")
    with insights_column:
        st.subheader("Live customer insights")
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Current visible detections", current_customers)
        metric_col2.metric("Active tracked customers", active_tracked_customers)
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Total unique customers", live_analytics["total_unique_customers"])
        metric_col2.metric("Estimated occupancy", (
            live_analytics["currently_inside"]
            if live_analytics["currently_inside"] is not None
            else "Unknown"
        ))
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Boundary entries", live_analytics["total_entries"])
        metric_col2.metric("Boundary exits", live_analytics["total_exits"])
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Peak visible count", st.session_state["_live_peak_customers"])
        metric_col2.metric(
            "Average observed duration",
            format_elapsed_time(average_dwell) if average_dwell is not None else "Collecting",
        )
        st.caption(f"Occupancy status: {live_analytics['occupancy_status']}.")
        st.write(f"**Longest observed track duration:** {format_elapsed_time(maximum_dwell)}.")
        st.write(
            f"**Crowd level:** {classify_crowd(current_customers, medium_crowd_threshold, high_crowd_threshold)[0]} "
            f"(average {live_analytics['average_crowd_level']})."
        )
        st.caption(
            f"Peak observed at {st.session_state['_live_peak_time']:.0f}s into this session."
        )
        st.write(f"**Traffic periods:** Busy: {busiest_period}; lower traffic: {quietest_period}.")
        st.write(f"**Frequently visited area:** {describe_hotspot(traffic_grid)}.")
        st.write(f"**Movement pattern:** {describe_movement(tracked)}.")
        if live_analytics["heatmap_position_samples"]:
            st.image(movement_heatmap, caption="Customer movement heatmap", use_container_width=True)
        else:
            st.info("Heatmap is waiting for confirmed tracked positions.")
        if st.session_state["_live_alerts"]:
            for alert in st.session_state["_live_alerts"]:
                st.warning(alert)
        live_export_records = add_session_metrics_to_records(
            st.session_state["_live_customer_records"],
            live_analytics,
        )
        live_csv = pd.DataFrame(live_export_records).to_csv(index=False)
        st.download_button(
            "Download customer session CSV",
            data=live_csv,
            file_name="customer-session-analytics.csv",
            mime="text/csv",
            key="live_customer_csv",
        )


st.set_page_config(page_title="Retail AI Insights", layout="wide")

st.markdown(
    """
    <style>
    :root {
        color-scheme: light;
        --text-color: #1f2937;
        --background-color: #ffffff;
        --secondary-background-color: #f5f8fc;
        --primary-color: #2563eb;
    }
    .stApp {
        background: #ffffff;
        color: #1f2937;
        font-family: "Segoe UI", Arial, sans-serif;
    }
    [data-testid="stAppViewContainer"] {
        background: #ffffff;
    }
    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #2563eb;
        letter-spacing: -0.02em;
    }
    h1 {
        padding-bottom: 0.6rem;
        border-bottom: 2px solid #bfdbfe;
    }
    [data-testid="stSidebar"] {
        background: #f5f8fc;
        border-right: 1px solid #dbe5f1;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stRadio"] label,
    [data-testid="stSidebar"] [data-testid="stCheckbox"] label {
        color: #334155 !important;
    }
    [data-testid="stWidgetLabel"],
    [data-testid="stCaptionContainer"],
    [data-testid="stMarkdownContainer"] p {
        color: #334155;
    }
    input, textarea, [data-baseweb="select"] > div {
        color: #1f2937 !important;
        background-color: #ffffff !important;
    }
    [data-testid="stMetric"] {
        height: 100%;
        padding: 1rem 1.1rem;
        background: #f8fafc;
        border: 1px solid #dbe5f1;
        border-left: 3px solid #2563eb;
        border-radius: 6px;
    }
    [data-testid="stMetricLabel"] {
        color: #475569;
        font-size: 0.88rem;
    }
    [data-testid="stMetricValue"] {
        color: #1f2937;
        font-size: 1.55rem;
    }
    .stButton > button {
        border-radius: 5px;
        font-weight: 600;
        border-color: #bfdbfe;
        color: #1d4ed8;
        background: #ffffff;
    }
    .stButton > button[kind="primary"] {
        background: #2563eb;
        border-color: #2563eb;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AI Retail Customer Experience Dashboard")

st.session_state.setdefault("_live_camera_capture", None)
st.session_state.setdefault("_live_camera_running", False)

with st.sidebar:
    st.header("Project settings")
    input_source = st.radio("Input source", ["Recorded Video", "IP Camera / Live Stream"])
    congestion_threshold = st.slider("Congestion threshold", min_value=2, max_value=20, value=8)
    low_traffic_threshold = st.slider("Low traffic threshold", min_value=0, max_value=5, value=1)
    medium_crowd_threshold = st.slider("Medium crowd from", min_value=2, max_value=19, value=5)
    high_crowd_threshold = st.slider(
        "High crowd from",
        min_value=medium_crowd_threshold + 1,
        max_value=30,
        value=max(12, medium_crowd_threshold + 1),
    )
    sound_alerts = st.checkbox("Enable sound alerts", value=True)
    st.caption("Important: This app analyzes anonymous movement patterns only and does not identify individuals.")

    uploaded_video = None
    analysis_button = False
    connect_stream = False
    stop_stream = False
    stream_url = ""
    if input_source == "Recorded Video":
        uploaded_video = st.file_uploader("Upload a recorded mall or retailer video", type=["mp4", "avi", "mov"])
        analysis_button = st.button("Start AI Analysis")
    else:
        stream_url = st.text_input(
            "IP camera / live-stream URL",
            value=os.environ.get("CAMERA_STREAM_URL", ""),
            type="password",
            placeholder="rtsp://user:password@camera-ip:554/stream",
            help="RTSP/RTSPS and HTTP/HTTPS (including MJPEG) streams are supported. Credentials are not logged or shown in connection errors.",
        )
        connect_stream = st.button("Connect Stream", type="primary")
        stop_stream = st.button("Disconnect Stream")

if input_source == "Recorded Video" and st.session_state["_live_camera_running"]:
    release_live_camera()

zone_names = DEFAULT_ZONES.copy()

if input_source == "IP Camera / Live Stream":
    if stop_stream:
        release_live_camera()
    if connect_stream:
        release_live_camera()
        try:
            capture, first_frame = open_ip_stream(stream_url)
            st.session_state["_live_camera_capture"] = capture
            st.session_state["_live_camera_running"] = True
            st.session_state["_live_pending_frame"] = first_frame
            st.session_state["_live_tracked"] = {}
            st.session_state["_live_next_person_id"] = 1
            st.session_state["_live_frame_index"] = 0
            st.session_state["_live_started_at"] = time.monotonic()
            st.session_state["_live_active_congested_zones"] = {}
            st.session_state["_live_alerts"] = []
            st.session_state["_live_customer_records"] = []
            st.session_state["_live_customer_records_by_id"] = {}
            st.session_state["_live_peak_customers"] = 0
            st.session_state["_live_peak_time"] = 0.0
            st.session_state["_live_activity_buckets"] = {}
            st.session_state["_live_crowd_levels"] = []
            st.session_state["_live_traffic_grid"] = np.zeros((30, 40), dtype=np.float32)
            st.session_state["_live_heatmap_position_samples"] = 0
        except ValueError as exc:
            st.error(str(exc))

    st.subheader("IP Camera / Live Stream")
    if st.session_state["_live_camera_running"]:
        render_live_camera(
            zone_names,
            congestion_threshold,
            low_traffic_threshold,
            sound_alerts,
            medium_crowd_threshold,
            high_crowd_threshold,
        )
    else:
        st.info("Enter an IP camera or live-stream URL and click Connect Stream to begin full-frame analysis.")

elif uploaded_video is not None and analysis_button:
    original_name = uploaded_video.name or "uploaded_video.mp4"
    file_suffix = Path(original_name).suffix.lower() or ".mp4"
    temp_video = OUTPUT_DIR / f"upload_{uuid.uuid4().hex}{file_suffix}"
    with open(temp_video, "wb") as fp:
        while chunk := uploaded_video.read(1024 * 1024):
            fp.write(chunk)

    progress_bar = st.progress(0, text="Starting video analysis...")
    progress_status = st.empty()

    def update_video_progress(progress, message):
        progress_bar.progress(min(100, int(progress * 100)), text=message)
        progress_status.caption(message)

    with st.spinner("Processing recorded video... This may take some time."):
        analysis = process_video(
            temp_video,
            zone_names,
            congestion_threshold,
            low_traffic_threshold,
            sound_enabled=sound_alerts,
            medium_crowd_threshold=medium_crowd_threshold,
            high_crowd_threshold=high_crowd_threshold,
            progress_callback=update_video_progress,
        )

    st.success("Analysis completed.")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total unique tracked IDs", analysis["summary"]["total_detected_customers"])
    col2.metric("Peak visible customers", analysis["summary"]["peak_customers"])
    col3.metric("Current visible customers", analysis["summary"]["current_customers"])
    col4.metric("Active tracked customers", analysis["summary"]["active_tracked_customers"])
    col5.metric("Boundary entries", analysis["summary"]["entry_count"])
    col6.metric("Boundary exits", analysis["summary"]["exit_count"])
    average_dwell = analysis["summary"]["average_dwell_time_seconds"]
    st.caption(
        f"Current visible count is the person detections on the latest analyzed frame; visible tracked counts use confirmed IDs. "
        f"Active tracks are retained for up to {TRACK_LOST_TIMEOUT_SECONDS:.1f} seconds after their last detection. "
        "Boundary entries/exits count confirmed inward/outward crossings of the full-frame edge band; "
        "people already in view at startup form an estimated initial occupancy."
    )
    st.subheader("Observed duration, occupancy and crowd summary")
    dwell_col1, dwell_col2, dwell_col3, dwell_col4 = st.columns(4)
    dwell_col1.metric("Average observed duration", format_elapsed_time(average_dwell))
    dwell_col2.metric("Maximum observed duration", format_elapsed_time(analysis["summary"]["maximum_dwell_time_seconds"]))
    dwell_col3.metric("Estimated occupancy", (
        analysis["summary"]["currently_inside"]
        if analysis["summary"]["currently_inside"] is not None
        else "Unknown"
    ))
    dwell_col4.metric("Average crowd level", analysis["summary"]["average_crowd_level"])
    st.write(
        f"**Occupancy status:** {analysis['summary']['occupancy_status']} "
        f"(initial estimate: {analysis['summary']['initial_occupancy_estimate']}; "
        "estimate = initial occupancy + valid entries − valid exits)."
    )
    st.write(f"**Maximum visible customers:** {analysis['summary']['maximum_crowd']}")
    st.write(f"**Boundary entries:** {analysis['summary']['entry_count']} · **Boundary exits:** {analysis['summary']['exit_count']}")
    st.write(f"**Peak period:** {analysis['summary']['peak_period']} · **Peak activity:** {analysis['summary']['peak_activity_time_seconds']:.1f}s")
    st.write(f"**Busy period:** {analysis['summary']['busiest_period']}.")
    st.write(f"**Lower-traffic period:** {analysis['summary']['quietest_period']}.")
    st.write(f"**Peak activity time:** {analysis['summary']['peak_activity_time_seconds']:.1f}s from clip start.")
    st.write(f"**Frequently visited area:** {analysis['summary']['most_visited_area']}.")
    st.write(f"**Movement pattern:** {analysis['summary']['movement_pattern']}.")
    st.write(f"**Full-frame traffic status:** {analysis['summary']['traffic_status']}.")
    st.caption("Observed duration is measured from first to last detection for each confirmed ID; it is not verified time inside the store. Peak time is relative to the clip and is not a business-hour estimate. Queue/waiting estimates are omitted because this full-frame tracker has no reliable queue-area signal.")

    st.subheader("Processed Video Preview")
    st.video(analysis["output_video"])

    st.subheader("Customer movement heatmap")
    if analysis["summary"]["heatmap_position_samples"]:
        st.image(analysis["heatmap_path"], use_container_width=True)
        st.caption(
            f"Built from {analysis['summary']['heatmap_position_samples']:,} confirmed track positions "
            "across analyzed frames."
        )
    else:
        st.info("No confirmed tracked positions were available to build a heatmap.")

    st.subheader("Full-frame customer activity")
    section_df = analysis["section_df"]
    activity_df = section_df.rename(
        columns={"Zone": "Area", "Visible Tracked Customers": "Visible Tracked Count"}
    )
    st.dataframe(
        activity_df[
            ["Area", "Visible Tracked Count", "Traffic Status", "Average Observed Duration", "Entries", "Exits"]
        ]
    )
    st.caption("Counts and observed duration are based on anonymous tracks. Estimated occupancy is marked unknown when unclassified arrivals or tracks lost without a boundary crossing make the event ledger incomplete.")
    if not section_df.empty:
        fig = px.bar(activity_df, x="Area", y="Visible Tracked Count", color="Visible Tracked Count", title="Visible tracked customers across the full frame")
        st.plotly_chart(fig, use_container_width=True)

        dwell_fig = px.bar(activity_df, x="Area", y="Avg Dwell (s)", color="Avg Dwell (s)", title="Average observed track duration across the full frame")
        st.plotly_chart(dwell_fig, use_container_width=True)

    st.subheader("Possible assistance opportunities")
    if analysis["assistance_opportunities"]:
        for item in analysis["assistance_opportunities"]:
            st.info(item)
    else:
        st.info("No specific assistance opportunity was identified from this video.")

    st.subheader("Congestion alerts")
    if analysis["alerts"]:
        for alert in analysis["alerts"]:
            st.warning(alert)
    else:
        st.info("No congestion alert triggered for the uploaded video.")

    st.subheader("Generated report")
    st.text(analysis["report"])

    st.subheader("Saved files")
    st.write("CSV data:", analysis["summary"]["csv_path"])
    customer_csv = pd.DataFrame(analysis["records"]).to_csv(index=False)
    st.download_button(
        "Download customer analytics CSV",
        data=customer_csv,
        file_name="customer-analytics.csv",
        mime="text/csv",
    )
    st.write("Report:", analysis["summary"]["report_path"])
    st.write("Video output:", analysis["summary"]["output_video"])

elif input_source == "Recorded Video" and uploaded_video is not None:
    st.info("Click 'Start AI Analysis' to process the selected video.")
elif input_source == "Recorded Video":
    st.info("Upload a recorded mall or retail video to begin analysis.")

st.markdown("""
---
This project analyzes anonymous customer movement patterns from recorded retail videos or an IP camera/live stream.
It does not identify individual customers by face or personal data.
""")
