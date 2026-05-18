from flask import Flask, Response, jsonify
from flask_cors import CORS
import threading
import cv2
import json
import queue

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

SCHEDULES = {
    "ejhay": {"hour": 13, "minute": 0}, 
}

GRACE_PERIOD_MINUTES = 15  

class StreamServer:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self.latest_data = {}
        self.sse_clients = []
        self.sse_lock = threading.Lock()

    def update_frame(self, frame):
        with self.lock:
            self.frame = frame.copy()

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def update_recognition(self, name, confidence, timestamp):
        if name == "Unknown":
            return

        from datetime import datetime
        now = datetime.now()

        if name in SCHEDULES:
            sched = SCHEDULES[name]
            sched_time = now.replace(
                hour=sched["hour"],
                minute=sched["minute"],
                second=0,
                microsecond=0
            )
            diff_minutes = (now - sched_time).total_seconds() / 60

            if diff_minutes < 0:
                status = "Early"
            elif diff_minutes <= GRACE_PERIOD_MINUTES:
                status = "On Time"
            else:
                status = f"Late by {int(diff_minutes)} mins"
        else:
            status = "No Schedule"

        data = {
            "name": name,
            "confidence": round(confidence * 100, 1),
            "time": timestamp,
            "status": status
        }

        self.latest_data = data
        self._push_sse(data)

    def _push_sse(self, data):
        message = f"data: {json.dumps(data)}\n\n"
        with self.sse_lock:
            for q in self.sse_clients:
                q.put(message)

    def subscribe(self):
        q = queue.Queue()
        with self.sse_lock:
            self.sse_clients.append(q)
        return q

    def unsubscribe(self, q):
        with self.sse_lock:
            self.sse_clients.remove(q)


stream_server = StreamServer()


def generate_stream():
    while True:
        frame = stream_server.get_frame()
        if frame is None:
            continue
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


def generate_sse(q):
    try:
        while True:
            message = q.get(timeout=30)
            yield message
    except:
        pass


@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/events')
def events():
    q = stream_server.subscribe()
    def stream():
        try:
            yield from generate_sse(q)
        finally:
            stream_server.unsubscribe(q)
    return Response(stream(), mimetype='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache',
                        'X-Accel-Buffering': 'no'
                    })


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


def start_server(host='0.0.0.0', port=5000):
    threading.Thread(
        target=lambda: app.run(host=host, port=port, threaded=True),
        daemon=True
    ).start()
    print(f"Stream server started at http://0.0.0.0:{port}/video_feed")
