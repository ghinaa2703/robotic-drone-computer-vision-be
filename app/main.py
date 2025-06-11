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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_streams = {}

def process_frame_blocking(frame, target_width, target_height, jpeg_quality):
    if frame is None:
        return None

    resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

    _, buffer = cv2.imencode('.jpg', resized_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

async def capture_and_stream_frames(
    websocket: WebSocket, 
    stream_url: str, 
    client_id: str,
    target_fps: int,
    jpeg_quality: int,
    resize_factor: float
):
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
    
    target_width = int(original_width * resize_factor)
    target_height = int(original_height * resize_factor)
    
    target_frame_duration = 1.0 / target_fps if target_fps > 0 else 0.01 
    
    logger.info(f"Targeting {target_width}x{target_height} resolution at {target_fps} FPS (sleep {target_frame_duration:.4f}s) with JPEG quality {jpeg_quality} (client {client_id}).")

    frame_counter = 0
    start_time = time.time()

    try:
        while True:
            if client_id not in active_streams or active_streams[client_id]['stop_event'].is_set():
                logger.info(f"Stop event set or client {client_id} entry removed. Stopping stream loop.")
                break

            ret, frame = await asyncio.to_thread(cap.read) 

            if not ret:
                logger.warning(f"End of stream for {stream_url} (client {client_id}). Looping if file.")
                if stream_url.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm', '.ogg')):
                    await asyncio.to_thread(lambda: cap.set(cv2.CAP_PROP_POS_FRAMES, 0))
                    ret, frame = await asyncio.to_thread(cap.read)
                    if not ret:
                        logger.error(f"Failed to loop video for client {client_id}. Breaking stream.")
                        break
                else:
                    logger.info(f"Live stream {stream_url} for client {client_id} ended. Breaking.")
                    break

            jpg_as_text = await asyncio.to_thread(
                process_frame_blocking, frame, target_width, target_height, jpeg_quality
            )

            if jpg_as_text is None:
                logger.warning(f"Processed frame was None for client {client_id}. Skipping send.")
                continue

            try:
                await websocket.send_json({
                    "type": "frame", 
                    "image": jpg_as_text,
                    "width": target_width,
                    "height": target_height,
                    "fps_target": target_fps
                })
                frame_counter += 1
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
        actual_fps = frame_counter / duration if duration > 0 else 0
        logger.info(f"Stream for client {client_id} processed {frame_counter} frames in {duration:.2f}s (Actual FPS: {actual_fps:.2f}).")

        if 'cap' in locals() and cap.isOpened():
            cap.release()
            logger.info(f"Stream resources for client {client_id} from {stream_url} released.")
        else:
            logger.info(f"Stream resources for client {client_id} already released or not opened.")

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
                resize_factor = message.get("resize_factor", 0.5)

                if not (0 < target_fps <= 60): target_fps = 20
                if not (0 < jpeg_quality <= 100): jpeg_quality = 50
                if not (0.1 <= resize_factor <= 1.0): resize_factor = 0.5


                if not stream_url:
                    await websocket.send_json({"type": "error", "message": "Stream URL is missing."})
                    continue

                current_task = active_streams[client_id]["task"]
                if current_task and not current_task.done():
                    logger.info(f"Client {client_id}: Stopping previous stream task.")
                    active_streams[client_id]["stop_event"].set()
                    try:
                        await asyncio.wait_for(current_task, timeout=2.0)
                    except asyncio.TimeoutError:
                        logger.warning(f"Client {client_id}: Previous stream task did not stop in time.")
                    finally:
                        active_streams[client_id]["stop_event"].clear()

                logger.info(f"Client {client_id}: Starting new stream task from {stream_url} with FPS={target_fps}, Quality={jpeg_quality}, Resize={resize_factor}")
                task = asyncio.create_task(
                    capture_and_stream_frames(websocket, stream_url, client_id, target_fps, jpeg_quality, resize_factor)
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