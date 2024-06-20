from flask import Flask, Response

app = Flask(__name__)

import cv2
import time
import ultralytics
from yolo_tflite import YoloTFLite
from yolo_onnx import YoloOnnx

from omegaconf import OmegaConf
from system_metrics import SystemMetricsCSV
from ultralytics import YOLO

def yolo_main(self, input):
    return self(input)[0].plot()[:, :, ::-1]
YOLO.main = yolo_main


MODELS = {
    'tflite': YoloTFLite,
    'ultralytics': lambda **kwargs: YOLO(kwargs['model']),
    'onnx' : YoloOnnx,
    
}



def gen_frames(target_fps=60):
    camera = cv2.VideoCapture(app.main_config['input'])  # Path to video file
    fps = 0
    frame_count = 0
    start_time = time.time()
    while True:
        success, frame = camera.read()
        if not success:
            break  # Stop if video ends

        frame_count += 1
        end_time = time.time()
        elapsed_time = end_time - start_time
        if elapsed_time >= 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        yield frame, fps

def frame_to_output(frame):
    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    return (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



def frames_loop(target_fps=30):
        data = gen_frames(target_fps)
        for frame, fps in data: 
            
            # res = DET_MODEL(frame)
            # app.system_metrics.write_metrics(fps)
            # yield frame_to_output(res[0].plot()[:, :, ::-1])
            yield  frame_to_output(app.model.main(frame))


@app.route('/video_feed')
def video_feed():
    return Response(frames_loop(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "Webcam Stream with FPS"


if __name__ == '__main__':
    config = OmegaConf.load('config.yml')
    app.system_metrics = SystemMetricsCSV(output_file=config['system_metrics_log_file'])  
    
    app.model = MODELS[config['model_type']](**config)

    app.main_config = config
    app.run(debug=True, host='0.0.0.0')
    
