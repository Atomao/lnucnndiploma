# Ultralytics YOLO 🚀, AGPL-3.0 license

import argparse

import cv2.dnn
import numpy as np


colors = np.random.uniform(0, 255, size=(100, 3))


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    Draws bounding boxes on the input image based on the provided arguments.

    Args:
        img (numpy.ndarray): The input image to draw the bounding box on.
        class_id (int): Class ID of the detected object.
        confidence (float): Confidence score of the detected object.
        x (int): X-coordinate of the top-left corner of the bounding box.
        y (int): Y-coordinate of the top-left corner of the bounding box.
        x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
        y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
    """
    label = f"det: ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


class YoloOpenCV:
    def __init__(self, model, **kwargs):
        self.model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(model)

    def main(self, input_image):
        """
        Main function to load ONNX model, perform inference, draw bounding boxes, and display the output image.
    
        Args:
            onnx_model (str): Path to the ONNX model.
            input_image (str): Path to the input image.
    
        Returns:
            list: List of dictionaries containing detection information such as class_id, class_name, confidence, etc.
        """
        # Load the ONNX model
        # Read the input image
        original_image = cv2.imread(input_image) if  not isinstance(input_image, np.ndarray) else input_image
        # [height, width, _] = original_image.shape
    
        # # Prepare a square image for inference
        # length = max((height, width))
        # image = np.zeros((length, length, 3), np.uint8)
        # image[0:height, 0:width] = original_image
    
        # # Calculate scale factor
        # scale = length / 640
    
        # # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(original_image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
        self.model.setInput(blob)
    
        # Perform inference
        outputs = self.model.forward()
        # pred = np.transpose(output[0])
    
        # # Prepare output array
        # x = (pred[:, 0] - pred[:, 2] / 2)
        # y = (pred[:, 1] - pred[:, 3] / 2) 
        # w = pred[:, 2] * x_factor
        # h = pred[:, 3] * y_factor
        # boxes = np.vstack([x, y, w, h]).T
        # class_ids = np.argmax(pred[:, 4:], axis=1)
        # scores = np.max(pred[:, 4:], axis=1)


        # # Apply NMS (Non-maximum suppression)
        # result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
    
        # detections = []
    
        # # Iterate through NMS results to draw bounding boxes and labels
        # for i in range(len(result_boxes)):
        #     index = result_boxes[i]
        #     box = boxes[index]
        #     draw_bounding_box(
        #         original_image,
        #         class_ids[index],
        #         scores[index],
        #         round(box[0] * scale),
        #         round(box[1] * scale),
        #         round((box[0] + box[2]) * scale),
        #         round((box[1] + box[3]) * scale),
        #     )
        return original_image
