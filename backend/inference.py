
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from auth_utils import get_current_user 
import threading
import os
import re
import uuid
import shutil
import sqlite3
import cv2
import numpy as np
from ultralytics import YOLO
import pysrt
from typing import Optional

router = APIRouter()

# --- Configuration ---
UPLOAD_DIR = "static/uploads"
PROCESSED_DIR = "static/processed"
DB_PATH = "users.db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

model = YOLO("models/best.pt")

MASK_CLASSES = {
    3: (0, 0, 255), 4: (0, 255, 0), 8: (255, 0, 0),
}

def sanitize_filename(filename: str) -> str:
    """Sanitizes a filename to be URL-safe."""
    filename = re.sub(r'[\\/*?:"<>|]', "", filename)
    filename = re.sub(r'\s+', '_', filename)
    return filename

def process_result_frame(result, img):
    """Helper function to draw masks and bounding boxes on a single frame."""
    if result.masks is not None:
        for mask, cls in zip(result.masks.data.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int)):
            if cls in MASK_CLASSES:
                color = MASK_CLASSES[cls]
                binary = (mask * 255).astype("uint8")
                binary_resized = cv2.resize(binary, (img.shape[1], img.shape[0]))
                colored_mask = np.zeros_like(img, dtype=np.uint8)
                for j in range(3):
                    colored_mask[..., j] = binary_resized * (color[j] / 255.0)
                img = cv2.addWeighted(img, 1.0, colored_mask, 0.4, 0)

    if result.boxes is not None:
        for box, cls_id in zip(result.boxes.data, result.boxes.cls):
            if int(cls_id) not in MASK_CLASSES:
                xyxy = box[:4].cpu().numpy().astype(int)
                conf = box[4].cpu().item()
                label = f"{model.names[int(cls_id)]} {conf:.2f}"
                hue = int(180 * int(cls_id) / len(model.names))
                color = tuple(int(c) for c in cv2.cvtColor(
                    np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0])
                cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                cv2.putText(img, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


# --- Job Status Store ---

job_status = {}
job_cancel_flags = {}
job_cancel_lock = threading.Lock()

def process_video_job(job_id, filename, upload_path, processed_path, current_user):
    try:
        cap = cv2.VideoCapture(upload_path)
        if not cap.isOpened():
            job_status[job_id] = {"status": "error", "detail": "Unable to open uploaded video."}
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(processed_path, fourcc, fps, (width, height))
        batch = []
        BATCH_SIZE = 4
        while True:
            with job_cancel_lock:
                if job_cancel_flags.get(job_id):
                    job_status[job_id] = {"status": "cancelled"}
                    break
            ret, frame = cap.read()
            if not ret:
                break
            batch.append(frame)
            if len(batch) == BATCH_SIZE:
                results = model.predict(batch, conf=0.4, iou=0.45, verbose=False)
                with job_cancel_lock:
                    if job_cancel_flags.get(job_id):
                        job_status[job_id] = {"status": "cancelled"}
                        break
                for result, img in zip(results, batch):
                    out.write(process_result_frame(result, img))
                batch = []
        if batch:
            with job_cancel_lock:
                if not job_cancel_flags.get(job_id):
                    results = model.predict(batch, conf=0.4, iou=0.45, verbose=False)
                    for result, img in zip(results, batch):
                        out.write(process_result_frame(result, img))
        cap.release()
        out.release()
        with job_cancel_lock:
            if job_cancel_flags.get(job_id):
                job_status[job_id] = {"status": "cancelled"}
            else:
                with sqlite3.connect(DB_PATH) as conn:
                    cursor = conn.cursor()
                    cursor.execute("CREATE TABLE IF NOT EXISTS videos (email TEXT, filename TEXT)")
                    cursor.execute("INSERT INTO videos (email, filename) VALUES (?, ?)", (current_user, filename))
                    conn.commit()
                job_status[job_id] = {
                    "status": "done",
                    "processed_video_url": f"/video/processed/{filename}"
                }
    except Exception as e:
        job_status[job_id] = {"status": "error", "detail": str(e)}
# Endpoint to cancel a running job
@router.post("/cancel/{job_id}")
def cancel_video_job(job_id: str):
    with job_cancel_lock:
        job_cancel_flags[job_id] = True
        job_status[job_id] = {"status": "cancelled"}
    return {"status": "cancelled"}

def parse_srt(srt_content: bytes):
    try:
        subs = pysrt.from_string(srt_content.decode('utf-8'))
        timeline = []
        for sub in subs:
            m = re.search(r"\[latitude\s*:\s*([\-\d\.]+)\].*?\[longtitude\s*:\s*([\-\d\.]+)\]", sub.text)
            if not m:
                continue
            lat, lon = float(m.group(1)), float(m.group(2))
            timeline.append({
                "start": sub.start.ordinal / 1000.0,
                "end":   sub.end.ordinal   / 1000.0,
                "lat": lat,
                "lon": lon
            })
        return timeline
    except Exception:
        return [] # Return empty list on parsing error


@router.post("/upload/")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    srt: Optional[UploadFile] = File(None), # SRT file is now optional
    current_user: str = Depends(get_current_user)
):
    sanitized_original_name = sanitize_filename(file.filename)
    filename = f"{uuid.uuid4()}_{sanitized_original_name}"
    upload_path = os.path.join(UPLOAD_DIR, filename)
    processed_path = os.path.join(PROCESSED_DIR, filename)

    # Save the uploaded video file
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Parse the SRT file if it exists
    timeline_data = []
    if srt:
        srt_content = await srt.read()
        timeline_data = parse_srt(srt_content)

    # Start the background video processing job
    job_id = str(uuid.uuid4())
    job_status[job_id] = {"status": "processing"}
    background_tasks.add_task(process_video_job, job_id, filename, upload_path, processed_path, current_user)

    # Return job_id for polling AND the timeline data for the map
    return {
        "job_id": job_id,
        "status": "processing",
        "timeline": timeline_data
    }

@router.get("/status/{job_id}")
def get_job_status(job_id: str):
    status = job_status.get(job_id)
    if not status:
        return {"status": "not_found"}
    return status

@router.get("/processed/{filename}")
def get_processed_video(filename: str):
    path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Processed video not found.")
    return FileResponse(path, media_type="video/mp4")

@router.get("/history")
def get_user_videos(current_user: str = Depends(get_current_user)):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS videos (email TEXT, filename TEXT)")
        cursor.execute("SELECT filename FROM videos WHERE email = ?", (current_user,))
        rows = cursor.fetchall()

    videos = [{
        "filename": row[0],
        "processed_url": f"/video/processed/{row[0]}",
    } for row in rows]

    return {"videos": videos}

@router.delete("/history/clear")
def clear_user_history(current_user: str = Depends(get_current_user)):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT filename FROM videos WHERE email = ?", (current_user,))
            filenames_to_delete = [row[0] for row in cursor.fetchall()]
            cursor.execute("DELETE FROM videos WHERE email = ?", (current_user,))
            conn.commit()

        for filename in filenames_to_delete:
            if os.path.exists(os.path.join(UPLOAD_DIR, filename)):
                os.remove(os.path.join(UPLOAD_DIR, filename))
            if os.path.exists(os.path.join(PROCESSED_DIR, filename)):
                os.remove(os.path.join(PROCESSED_DIR, filename))
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

