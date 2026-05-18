from flask import Flask, Response
import threading
import cv2

app = Flask(__name__)

class StreamServer:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()

    def update_frame(self, frame):
        with self.lock:
            self.frame = frame.copy()

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

stream_server = StreamServer()

def generate():
    while True:
        frame = stream_server.get_frame()
        if frame is None:
            continue
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health')
def health():
    return {'status': 'ok'}

def start_server(host='0.0.0.0', port=5000):
    threading.Thread(
        target=lambda: app.run(host=host, port=port, threaded=True),
        daemon=True
    ).start()
    print(f"Stream server started at http://0.0.0.0:{port}/video_feed")
