from flask import Flask, Response

app = Flask(__name__)

import cv2
import time
import ultralytics
from yolo_tflite import YoloTFLite
from yolo_onnx import YoloOnnx
from yolo_opencv import YoloOpenCV

from omegaconf import OmegaConf
from system_metrics import SystemMetricsCSV
from ultralytics import YOLO

import cv2
import time
import os
import psutil
import shutil
from ultralytics import YOLO  # Ensure this is the correct import for your YOLO implementation
import csv


def yolo_main(self, input):
    return self(input)[0].plot()[:, :, ::-1]
YOLO.main = yolo_main


MODELS = {
    'tflite': YoloTFLite,
    'ultralytics': lambda **kwargs: YOLO(kwargs['model']),
    'onnx' : YoloOnnx,
    'opencv': YoloOpenCV
    
}






def get_cpu_usage():
    return psutil.cpu_percent(interval=1)

def get_ram_usage():
    ram = psutil.virtual_memory()
    return ram.used / (1024 * 1024)  # Convert to MB

def get_system_temperature():
    # This method of getting temperature works on Raspberry Pi.
    # You might need to modify this according to your system.
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as temp_file:
            cpu_temp = int(temp_file.read()) / 1000.0
        return cpu_temp
    except FileNotFoundError:
        return -1  # Return -1 if temperature reading is not available

def run_inference_on_folder(input_folder, output_folder, config):
    model = MODELS[config['model_type']](**config)
    system_metrics = SystemMetricsCSV(output_file=config['system_metrics_log_file'])  
    # Ensure output directory exists
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)  # Remove the output folder if it exists
    os.makedirs(output_folder)

    # Prepare CSV file for system metrics
    csv_file = open('system_metrics.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Filename', 'Inference Time (ms)', 'CPU Usage (%)', 'RAM Usage (MB)', 'Temperature (C)'])

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_folder, filename)
            frame = cv2.imread(file_path)

            start_time = time.time()
            annotated_image = model.main(frame)
            inference_time = (time.time() - start_time) * 1000  # Inference time in milliseconds

            print(filename, inference_time)
            # Optionally save the annotated image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, annotated_image)

            system_metrics.write_metrics(filename, inference_time)

    csv_file.close()


if __name__ == '__main__':
    config = OmegaConf.load('config.yml')
    input_folder = "/tmp/test/video"
    output_folder = "/tmp/test/video_output/";
    run_inference_on_folder(input_folder, output_folder, config)
