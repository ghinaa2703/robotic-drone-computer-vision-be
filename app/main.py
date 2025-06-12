from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import base64
import asyncio
import numpy as np
import json
import logging
import time
import os
import torch
import warnings
import pickle
import face_recognition 
from ultralytics import YOLO 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "http://localhost:3000",  
    "http://127.0.0.1:3000", 
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

yolo_model: YOLO = None 
known_face_encodings: list = []
known_face_names: list = []
FACE_ENCODINGS_CACHE_PATH = 'face_encodings_cache.pkl' 
FACE_DATASET_PATH = 'dataset/face_recognition'      

active_streams = {} 
client_processing_state = {}

def get_dataset_last_modified(dataset_path):
    last_modified = 0
    if not os.path.exists(dataset_path): 
        return 0 
    for root, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            mtime = os.path.getmtime(file_path)
            if mtime > last_modified:
                last_modified = mtime
    return last_modified

def load_known_faces_blocking(dataset_path, use_cache=True, cache_path='face_encodings_cache.pkl'):
    if use_cache and os.path.exists(cache_path):
        try:
            logger.info(f"Attempting to load face encodings from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            if cache_data.get('dataset_path') == dataset_path:
                dataset_mtime = get_dataset_last_modified(dataset_path)
                if dataset_mtime <= cache_data.get('last_modified', 0):
                    logger.info(f"Cache valid. Loaded {len(cache_data['encodings'])} faces from cache.")
                    return cache_data['encodings'], cache_data['names']
                else:
                    logger.info("Dataset changed since cache was made. Rebuilding cache.")
            else:
                logger.info("Cache found but for a different dataset path. Rebuilding cache.")
        except Exception as e:
            logger.warning(f"Error reading cache: {e}. Rebuilding cache.")
    
    logger.info("Loading known faces from dataset (no valid cache found or cache disabled)...")
    known_encodings = []
    known_names = []
    
    if not os.path.exists(dataset_path):
        logger.warning(f"Face dataset directory not found at {dataset_path}. No faces will be recognized.")
        return known_encodings, known_names

    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                try:
                    face_image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(face_image)
                    
                    if face_encodings:
                        known_encodings.append(face_encodings[0])
                        known_names.append(person_name)
                    else:
                        logger.warning(f"  - No face found in {image_path}")
                except Exception as e:
                    logger.error(f"  - Error processing {image_path}: {e}")
                    
    logger.info(f"Total {len(known_encodings)} known faces loaded from dataset.")
    
    if use_cache and known_encodings:
        try:
            cache_data = {
                'dataset_path': dataset_path,
                'last_modified': get_dataset_last_modified(dataset_path),
                'encodings': known_encodings,
                'names': known_names
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Face encodings successfully saved to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    return known_encodings, known_names

def encode_frame_to_jpg_blocking(frame: np.ndarray, target_width: int, target_height: int, jpeg_quality: int) -> str:
    if frame is None:
        return None
    resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

    _, buffer = cv2.imencode('.jpg', resized_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text
def process_cv_frame_blocking(
    frame: np.ndarray,
    client_state: dict,
    yolo_model_instance: YOLO,
    known_face_encodings_list: list,
    known_face_names_list: list,
    conf_threshold: float,
    iou_threshold: float,
    cv_processing_resize_factor: float,
    face_detection_interval: int,
    crowd_threshold: int,
    high_density_threshold: int
) -> dict:
    
    if frame is None:
        return {"annotated_frame": None, "metadata": None}

    frame_count = client_state.get("frame_count", 0)
    face_locations = client_state.get("face_locations", [])
    face_names = client_state.get("face_names", [])
    face_confidence = client_state.get("face_confidence", [])

    height, width, _ = frame.shape

    input_frame_for_cv = frame 
    if cv_processing_resize_factor != 1.0:
        input_frame_for_cv = cv2.resize(
            frame, 
            (int(width * cv_processing_resize_factor), int(height * cv_processing_resize_factor)), 
            interpolation=cv2.INTER_AREA
        )
    
    yolo_results = yolo_model_instance.predict(
        input_frame_for_cv,
        conf=conf_threshold, 
        iou=iou_threshold, 
        classes=0,
        verbose=False
    )
    
    person_boxes = yolo_results[0].boxes
    person_count = len(person_boxes)

    grid_size = 3
    grid_width = width // grid_size
    grid_height = height // grid_size
    grid_counts = np.zeros((grid_size, grid_size), dtype=int)

    for box in person_boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        
        if cv_processing_resize_factor != 1.0:
            x1 = int(x1 / cv_processing_resize_factor)
            y1 = int(y1 / cv_processing_resize_factor)
            x2 = int(x2 / cv_processing_resize_factor)
            y2 = int(y2 / cv_processing_resize_factor)
        
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        grid_x = min(max(0, center_x // grid_width), grid_size - 1)
        grid_y = min(max(0, center_y // grid_height), grid_size - 1)
        grid_counts[grid_y, grid_x] += 1
    
    is_crowd_bool = person_count >= crowd_threshold
    is_high_density_bool = person_count >= high_density_threshold or np.max(grid_counts) >= (high_density_threshold // grid_size)
    is_crowd = bool(is_crowd_bool)
    is_high_density = bool(is_high_density_bool) 

    if frame_count % face_detection_interval == 0:
        rgb_input_for_face_recognition = cv2.cvtColor(input_frame_for_cv, cv2.COLOR_BGR2RGB) 
        
        current_face_locations = face_recognition.face_locations(rgb_input_for_face_recognition)
        current_face_encodings = face_recognition.face_encodings(rgb_input_for_face_recognition, current_face_locations)
        
        face_names_current = []
        face_confidence_current = []
        
        for current_encoding in current_face_encodings:
            name = "Tidak Dikenal"
            confidence = 0.0
            
            if known_face_encodings_list:
                face_distances = face_recognition.face_distance(known_face_encodings_list, current_encoding)
                best_match_index = np.argmin(face_distances)
                best_distance = face_distances[best_match_index]
                
                if best_distance < 0.6:
                    name = known_face_names_list[best_match_index]
                    confidence = round((1 - best_distance) * 100, 1)
            
            face_names_current.append(name)
            face_confidence_current.append(confidence)
        
        face_locations_scaled = []
        if cv_processing_resize_factor != 1.0:
            for (top, right, bottom, left) in current_face_locations:
                top_s = int(top / cv_processing_resize_factor)
                right_s = int(right / cv_processing_resize_factor)
                bottom_s = int(bottom / cv_processing_resize_factor)
                left_s = int(left / cv_processing_resize_factor)
                face_locations_scaled.append((top_s, right_s, bottom_s, left_s))
        else:
            face_locations_scaled = current_face_locations 
            
        face_locations = face_locations_scaled 
        face_names = face_names_current
        face_confidence = face_confidence_current
        client_state["face_locations"] = face_locations
        client_state["face_names"] = face_names
        client_state["face_confidence"] = face_confidence
    else:
        pass 

    annotated_frame = frame.copy()

    for y_grid in range(grid_size):
        for x_grid in range(grid_size):
            count = grid_counts[y_grid, x_grid]
            if count > 0:
                if count >= (high_density_threshold // grid_size):
                    grid_color = (0, 0, 255)
                    grid_alpha = 0.5
                elif count >= (crowd_threshold // grid_size):
                    grid_color = (0, 165, 255)
                    grid_alpha = 0.3
                else:
                    grid_color = (0, 255, 255) 
                    grid_alpha = 0.2
                
                overlay = annotated_frame.copy()
                cv2.rectangle(
                    overlay, 
                    (x_grid * grid_width, y_grid * grid_height), 
                    ((x_grid + 1) * grid_width, (y_grid + 1) * grid_height), 
                    grid_color, 
                    -1 
                )
                cv2.addWeighted(overlay, grid_alpha, annotated_frame, 1 - grid_alpha, 0, annotated_frame)
                
                cv2.rectangle(
                    annotated_frame, 
                    (x_grid * grid_width, y_grid * grid_height), 
                    ((x_grid + 1) * grid_width, (y_grid + 1) * grid_height), 
                    grid_color, 
                    2
                )
                
                cv2.putText(
                    annotated_frame,
                    str(count),
                    (x_grid * grid_width + grid_width // 2 - 10, y_grid * grid_height + grid_height // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
    
    for (top, right, bottom, left), name, conf in zip(face_locations, face_names, face_confidence):
        box_color = (0, 255, 0) if name != "Tidak Dikenal" else (0, 0, 255)
        cv2.rectangle(annotated_frame, (left, top), (right, bottom), box_color, 2)
        
        label = f"{name} ({conf:.1f}%)" if conf > 0 else name
        label_y = max(top - 15, 10)
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        
        cv2.rectangle(
            annotated_frame, 
            (left, label_y - text_size[1] - 5), 
            (left + text_size[0] + 5, label_y + 5), 
            box_color, 
            cv2.FILLED
        )
        cv2.putText(
            annotated_frame, 
            label, 
            (left + 3, label_y), 
            font, 
            font_scale, 
            (255, 255, 255),
            font_thickness
        )

    for box in person_boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        
        if cv_processing_resize_factor != 1.0:
            x1 = int(x1 / cv_processing_resize_factor)
            y1 = int(y1 / cv_processing_resize_factor)
            x2 = int(x2 / cv_processing_resize_factor)
            y2 = int(y2 / cv_processing_resize_factor)
        
        is_covered_by_face_box = False
        for (f_top, f_right, f_bottom, f_left) in face_locations:
            overlap_x = max(0, min(x2, f_right) - max(x1, f_left))
            overlap_y = max(0, min(y2, f_bottom) - max(y1, f_top))
            overlap_area = overlap_x * overlap_y
            face_area_actual = (f_right - f_left) * (f_bottom - f_top)

            if overlap_area > 0 and face_area_actual > (0.1 * ((x2 - x1) * (y2 - y1))):
                is_covered_by_face_box = True
                break
        
        if not is_covered_by_face_box:
            conf = float(box.conf[0]) * 100
            box_color = (255, 0, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
            
            label = f"Person ({conf:.1f}%)"
            label_y = max(y1 - 10, 10)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            cv2.rectangle(
                annotated_frame,
                (x1, label_y - text_size[1] - 5),
                (x1 + text_size[0] + 5, label_y + 5),
                box_color,
                cv2.FILLED
            )
            cv2.putText(
                annotated_frame,
                label,
                (x1 + 3, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

    status_text_banner = "NORMAL"
    status_color_banner = (0, 255, 0)
    crowd_status_message_banner = ""

    if is_high_density:
        status_text_banner = "KERUMUNAN PADAT!"
        status_color_banner = (0, 0, 255) 
        dense_zones = []
        for y_grid in range(grid_size):
            for x_grid in range(grid_size):
                if grid_counts[y_grid, x_grid] >= (high_density_threshold // grid_size):
                    dense_zones.append(f"({x_grid+1},{y_grid+1})")
        if dense_zones:
            crowd_status_message_banner = f"DENSE: {', '.join(dense_zones)}"
    elif is_crowd:
        status_text_banner = "KERUMUNAN TERDETEKSI"
        status_color_banner = (0, 165, 255) 
        crowd_status_message_banner = "CROWD DETECTED"
    
    cv2.rectangle(annotated_frame, (0, 0), (width, 40), (0, 0, 0), -1) 
    cv2.putText(annotated_frame, f"Orang: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"Status: {status_text_banner} {crowd_status_message_banner}", (width // 2 - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color_banner, 2)

    client_state["frame_count"] = frame_count + 1

    return {
        "annotated_frame": annotated_frame,
        "metadata": {
            "person_count": person_count,
            "density_zones": grid_counts.tolist(),
            "is_crowd": is_crowd,
            "is_high_density": is_high_density,
            "recognized_faces": [{"name": n, "confidence": c} for n, c in zip(face_names, face_confidence)],
        }
    }
@app.on_event("startup")
async def startup_event():
    global yolo_model, known_face_encodings, known_face_names 

    logger.info("Application starting up. Loading models and face data...")
    yolo_model_path = 'yolov8n.pt'
    
    device = '0' 
    if not torch.cuda.is_available():
        device = 'cpu'
        logger.warning("CUDA is not available. Using CPU for YOLO processing. This will be significantly slower.")
    else:
        logger.info(f"CUDA is available. Using GPU device: {device} for YOLO processing.")
    
    logger.info(f"Loading YOLO model from {yolo_model_path} on device: {device}")
    yolo_model_loaded = await asyncio.to_thread(lambda: YOLO(yolo_model_path).to(device)) 
    yolo_model = yolo_model_loaded 
    logger.info("YOLO model loaded.")

    logger.info(f"Loading known faces from {FACE_DATASET_PATH}...")
    known_face_encodings_loaded, known_face_names_loaded = await asyncio.to_thread(
        load_known_faces_blocking, FACE_DATASET_PATH, use_cache=True, cache_path=FACE_ENCODINGS_CACHE_PATH
    )
    known_face_encodings = known_face_encodings_loaded 
    known_face_names = known_face_names_loaded 
    logger.info(f"Loaded {len(known_face_names)} known faces.")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down. Cleaning up active streams...")
    for client_id, state in list(active_streams.items()): 
        if state['task'] and not state['task'].done():
            state['stop_event'].set() 
            try:
                await asyncio.wait_for(state['task'], timeout=1.0) 
            except asyncio.TimeoutError:
                logger.warning(f"Stream task for client {client_id} did not stop gracefully during shutdown.")
        if client_id in active_streams: 
            del active_streams[client_id]
    logger.info("All active streams cleaned up.")

async def capture_and_stream_frames(
    websocket: WebSocket, 
    stream_url: str, 
    client_id: str,
    target_fps: int,
    jpeg_quality: int, 
    output_resize_factor: float,
    cv_processing_resize_factor: float 
):
    global yolo_model, known_face_encodings, known_face_names 

    if yolo_model is None or not known_face_encodings or not known_face_names:
        logger.error(f"Models or face data not loaded for client {client_id}. Cannot start streaming.")
        try:
            await websocket.send_json({"type": "error", "message": "Server models not ready. Please check server logs."})
        except (WebSocketDisconnect, RuntimeError):
            logger.warning(f"Could not send server error to client {client_id} (WebSocket already closed).")
        return
    
    if client_id not in client_processing_state:
        client_processing_state[client_id] = {
            "frame_count": 0,
            "face_locations": [],
            "face_names": [],
            "face_confidence": []
        }
    
    conf_threshold = 0.45 
    iou_threshold = 0.45 
    crowd_threshold = 5 
    high_density_threshold = 10
    face_detection_interval = 5 

    cap = await asyncio.to_thread(cv2.VideoCapture, stream_url) 
    
    if not cap.isOpened():
        logger.error(f"Failed to open stream from URL: {stream_url} for client {client_id}")
        try:
            await websocket.send_json({"type": "error", "message": f"Could not open stream: {stream_url}"})
        except (WebSocketDisconnect, RuntimeError):
            logger.warning(f"Could not send stream open error to client {client_id} (WebSocket already closed).")
        return

    logger.info(f"Successfully opened stream {stream_url} for client {client_id}. Starting frame capture loop.")

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Original stream resolution: {original_width}x{original_height} (client {client_id})")
    
    target_frame_duration = 1.0 / target_fps if target_fps > 0 else 0.01 
    
    logger.info(f"Processing stream for client {client_id} with: TARGET FPS={target_fps}, JPEG Quality={jpeg_quality}, Output Resize Factor={output_resize_factor}")
    logger.info(f"CV Processing will use: YOLO Confidence={conf_threshold}, Face Detection Interval={face_detection_interval}, CV Input Resize Factor={cv_processing_resize_factor}")

    frame_counter_actual = 0 
    start_time = time.time() 

    try:
        while True:
            if client_id not in active_streams or active_streams[client_id]['stop_event'].is_set():
                logger.info(f"Stop event set or client {client_id} entry removed. Stopping stream loop.")
                break

            ret, frame = await asyncio.to_thread(cap.read) 

            if not ret:
                logger.warning(f"End of stream for {stream_url} (client {client_id}). Looping if file, else breaking.")
                if stream_url.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm', '.ogg')):
                    await asyncio.to_thread(lambda: cap.set(cv2.CAP_PROP_POS_FRAMES, 0))
                    ret, frame = await asyncio.to_thread(cap.read) 
                    if not ret:
                        logger.error(f"Failed to loop video for client {client_id}. Breaking stream.")
                        break
                else:
                    logger.info(f"Live stream {stream_url} for client {client_id} ended. Breaking.")
                    break

            cv_results = await asyncio.to_thread(
                process_cv_frame_blocking, 
                frame,
                client_processing_state[client_id],
                yolo_model, 
                known_face_encodings,
                known_face_names,
                conf_threshold,
                iou_threshold,
                cv_processing_resize_factor,
                face_detection_interval,
                crowd_threshold,
                high_density_threshold
            )
            
            annotated_frame = cv_results["annotated_frame"]
            metadata = cv_results["metadata"]

            if annotated_frame is None:
                logger.warning(f"Annotated frame was None for client {client_id}. Skipping send.")
                continue 
            
            final_output_width = int(original_width * output_resize_factor)
            final_output_height = int(original_height * output_resize_factor)
            
            jpg_as_text = await asyncio.to_thread(
                encode_frame_to_jpg_blocking,
                annotated_frame, 
                final_output_width, 
                final_output_height, 
                jpeg_quality
            )

            if jpg_as_text is None:
                logger.warning(f"Final JPEG encoding failed for client {client_id}. Skipping send.")
                continue 

            try:
                await websocket.send_json({
                    "type": "frame", 
                    "image": jpg_as_text,
                    "width": final_output_width,  
                    "height": final_output_height,
                    "fps_target": target_fps,
                    "metadata": metadata  
                })
                frame_counter_actual += 1 
            except (WebSocketDisconnect, RuntimeError) as e:
                logger.warning(f"Failed to send frame to client {client_id} (WebSocket closed/runtime error): {e}")
                break 

            await asyncio.sleep(target_frame_duration)


    except Exception as e:
        logger.error(f"An unexpected error occurred during streaming for {client_id}: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": f"Streaming error: {e}"})
        except (WebSocketDisconnect, RuntimeError):
            logger.warning(f"Could not send final error to client {client_id} (WebSocket already closed).")
    finally:
        end_time = time.time()
        duration = end_time - start_time
        actual_fps = frame_counter_actual / duration if duration > 0 else 0
        logger.info(f"Stream for client {client_id} processed {frame_counter_actual} frames in {duration:.2f}s (Actual FPS: {actual_fps:.2f}).")

        if 'cap' in locals() and cap.isOpened():
            cap.release()
            logger.info(f"Stream resources for client {client_id} from {stream_url} released.")
        else:
            logger.info(f"Stream resources for client {client_id} already released or not opened.")
        
        if client_id in client_processing_state:
            del client_processing_state[client_id]
            logger.info(f"Client {client_id} processing state cleaned up.")


@app.websocket("/ws/video_stream")
async def video_stream_websocket(websocket: WebSocket):
    await websocket.accept()
    client_id = f"{websocket.client.host}:{websocket.client.port}" 
    logger.info(f"Client {client_id} connected to /ws/video_stream")

    active_streams[client_id] = {"stop_event": asyncio.Event(), "task": None}

    try:
        while True:
            message = await websocket.receive_json()
            message_type = message.get("type")

            if message_type == "start_stream":
                stream_url = message.get("url")
                target_fps = message.get("fps", 20)
                jpeg_quality = message.get("quality", 50)
                output_resize_factor = message.get("resize_factor", 0.5) 
                cv_processing_resize_factor = message.get("cv_processing_resize", 0.5) 


                if not (0 < target_fps <= 60): target_fps = 20
                if not (0 < jpeg_quality <= 100): jpeg_quality = 50
                if not (0.1 <= output_resize_factor <= 1.0): output_resize_factor = 0.5
                if not (0.1 <= cv_processing_resize_factor <= 1.0): cv_processing_resize_factor = 0.5

                if not stream_url:
                    await websocket.send_json({"type": "error", "message": "Stream URL is missing."})
                    continue

                current_task = active_streams[client_id]["task"]
                if current_task and not current_task.done():
                    logger.info(f"Client {client_id}: Stopping previous stream task to restart with new parameters.")
                    active_streams[client_id]["stop_event"].set()
                    try:
                        await asyncio.wait_for(current_task, timeout=2.0)
                    except asyncio.TimeoutError:
                        logger.warning(f"Client {client_id}: Previous stream task did not stop in time.")
                    finally:
                        active_streams[client_id]["stop_event"].clear()

                logger.info(f"Client {client_id}: Starting new stream task from {stream_url} with FPS={target_fps}, Quality={jpeg_quality}, Output Resize={output_resize_factor}, CV Resize={cv_processing_resize_factor}")
                task = asyncio.create_task(
                    capture_and_stream_frames(
                        websocket, stream_url, client_id, 
                        target_fps, jpeg_quality, output_resize_factor, cv_processing_resize_factor
                    )
                )
                active_streams[client_id]["task"] = task 

            elif message_type == "stop_stream":
                current_task = active_streams[client_id]["task"]
                if current_task and not current_task.done():
                    logger.info(f"Client {client_id}: Stopping active stream by explicit request.")
                    active_streams[client_id]["stop_event"].set() 
                    await websocket.send_json({"type": "status", "message": "Stream stopped."}) 
                    try:
                        await asyncio.wait_for(current_task, timeout=2.0) 
                    except asyncio.TimeoutError:
                        logger.warning(f"Client {client_id}: Stop stream task did not complete in time on explicit stop.")
                    finally:
                        active_streams[client_id]["stop_event"].clear()
                        active_streams[client_id]["task"] = None 
                else:
                    logger.info(f"Client {client_id}: No active stream to stop on explicit request.")
                    await websocket.send_json({"type": "status", "message": "No active stream to stop."})

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected from /ws/video_stream (main loop).")
    except Exception as e:
        logger.error(f"Unexpected error in main WebSocket loop for client {client_id}: {e}", exc_info=True)
    finally:
        if client_id in active_streams:
            current_task = active_streams[client_id]["task"]
            if current_task and not current_task.done():
                logger.info(f"Client {client_id}: Cleaning up still-running stream task on disconnect.")
                active_streams[client_id]["stop_event"].set() 
                try:
                    await asyncio.wait_for(current_task, timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning(f"Client {client_id}: Stream task did not stop in time during final cleanup.")
            
            del active_streams[client_id] 
            logger.info(f"Client {client_id} cleaned up from active_streams.")
        else:
            logger.info(f"Client {client_id} was already cleaned up or never fully initialized in active_streams.")

        try:
            await websocket.close()
        except RuntimeError as e:
            if "Cannot call \"send\" once a close message has been sent." in str(e):
                logger.warning(f"Client {client_id}: WebSocket already closed or in closing state. Ignoring RuntimeError.")
            else:
                raise e
        except Exception as e:
            logger.error(f"Client {client_id}: Error during final WebSocket close: {e}", exc_info=True)
        
        logger.info(f"Connection for client {client_id} fully closed.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)