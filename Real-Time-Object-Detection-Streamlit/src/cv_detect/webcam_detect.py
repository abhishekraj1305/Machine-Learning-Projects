import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(
            f"Config file '{path}' is empty or invalid YAML. "
            "Please fill configs/default.yaml with valid YAML content."
        )
    if not isinstance(data, dict):
        raise ValueError(f"Config file '{path}' must contain a YAML mapping (key: value).")
    return data


def open_camera(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # CAP_DSHOW is often more stable on Windows
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {index}. "
            f"Try index=1 or close apps like Teams/Zoom using the camera."
        )
    return cap


def get_class_name_map(model: YOLO) -> dict:
    # Ultralytics YOLO model stores class names in model.names
    names = model.names
    if isinstance(names, dict):
        return names
    # sometimes it's a list
    return {i: n for i, n in enumerate(names)}


def allowed_class_ids(names_map: dict, class_filter: list[str]) -> set[int] | None:
    if not class_filter:
        return None
    wanted = {c.strip().lower() for c in class_filter}
    allowed = set()
    for cid, cname in names_map.items():
        if str(cname).lower() in wanted:
            allowed.add(int(cid))
    return allowed


def should_alert(names_map: dict, alert_classes: list[str], detected_class_ids: set[int]) -> bool:
    if not alert_classes:
        return False
    wanted = {c.strip().lower() for c in alert_classes}
    detected_names = {str(names_map[cid]).lower() for cid in detected_class_ids}
    return len(wanted.intersection(detected_names)) > 0


def draw_overlay(frame, fps: float, inf_ms: float, alert_on: bool):
    h, w = frame.shape[:2]
    text1 = f"FPS: {fps:.1f}"
    text2 = f"Inference: {inf_ms:.1f} ms"
    cv2.putText(frame, text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, text2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    if alert_on:
        banner_h = 50
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 255), -1)
        alpha = 0.35
        frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        cv2.putText(frame, "ALERT: target object detected!", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def main(config_path: str = "configs/default.yaml"):
    cfg = load_config(config_path)

    model_path = cfg.get("model", "yolov8n.pt")
    imgsz = int(cfg.get("imgsz", 640))
    conf = float(cfg.get("conf", 0.35))
    iou = float(cfg.get("iou", 0.45))
    cam_index = int(cfg.get("camera_index", 0))

    class_filter = cfg.get("class_filter", []) or []
    alert_classes = cfg.get("alert_classes", []) or []

    save_video = bool(cfg.get("save_video", True))
    output_path = Path(cfg.get("output_path", "outputs/detections.mp4"))
    show_window = bool(cfg.get("show_window", True))
    max_fps = float(cfg.get("max_fps", 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    names_map = get_class_name_map(model)
    allowed_ids = allowed_class_ids(names_map, class_filter)

    cap = open_camera(cam_index)

    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        fps_out = 25.0
        writer = cv2.VideoWriter(str(output_path), fourcc, fps_out, (width, height))

    prev_time = time.perf_counter()
    fps_smooth = 0.0

    try:
        while True:
            if max_fps and max_fps > 0:
                time.sleep(1.0 / max_fps)

            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed. Exiting.")
                break

            # Inference
            t0 = time.perf_counter()
            results = model.predict(
                source=frame,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                verbose=False
            )
            inf_ms = (time.perf_counter() - t0) * 1000.0

            r0 = results[0]
            annotated = r0.plot()  # BGR annotated frame

            # Determine detected classes
            detected_ids = set()
            if r0.boxes is not None and len(r0.boxes) > 0:
                cls = r0.boxes.cls.detach().cpu().numpy().astype(int)
                for cid in cls.tolist():
                    if (allowed_ids is None) or (cid in allowed_ids):
                        detected_ids.add(cid)

            alert_on = should_alert(names_map, alert_classes, detected_ids)

            # FPS calc
            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            fps = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = fps if fps_smooth == 0 else (0.85 * fps_smooth + 0.15 * fps)

            draw_overlay(annotated, fps_smooth, inf_ms, alert_on)

            if writer is not None:
                writer.write(annotated)

            if show_window:
                cv2.imshow("CV Webcam Object Detection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or ESC
                    break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if save_video:
        print(f"Saved output video to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
