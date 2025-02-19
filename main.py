from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import base64
import cv2
import numpy as np
import uvicorn

app = FastAPI()

background_subtractor = cv2.createBackgroundSubtractorMOG2()

@app.get('/health')
def health_check():
    return {'status': 'healthy'}

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket Connection Established")

    try:
        while True:
            data = await websocket.receive_text()
            # Decode base64 image
            encoded_data = data.split(",")[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Convert to grayscale for motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fg_mask = background_subtractor.apply(gray)

            # Find contours (detect motion)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_detected = any(cv2.contourArea(contour) > 500 for contour in contours)

            # Send motion alert to frontend
            if motion_detected:
                await websocket.send_text("motion_detected")

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Check if the WebSocket connection is still open before closing
        if websocket.client_state == "connected":
            await websocket.close()
            print("WebSocket connection closed gracefully")

# 🔹 Run FastAPI with SSL
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, ssl_keyfile="C:\\WINDOWS\\system32\\private.key",
            ssl_certfile="C:\\WINDOWS\\system32\\certificate.crt")