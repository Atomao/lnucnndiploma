import cv2
import numpy as np
from tflite_runtime import interpreter as tflite
from letterbox import LetterBox


class YoloTFLite:
    def __init__(self, model, size, confidence_thres, iou_thres, num_threads, **kwargs):
        self.size = size
        self.tflite_model = model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        interpreter = tflite.Interpreter(model_path=self.tflite_model, 
                                        num_threads=num_threads)
        self.model = interpreter
        self.model.allocate_tensors()

        # Load the class names from the COCO dataset

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(100, 3))

    def draw_detections(self, img, box, score, class_id):
        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"det: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img,
            (int(label_x), int(label_y - label_height)),
            (int(label_x + label_width), int(label_y + label_height)),
            color,
            cv2.FILLED,
        )

        # Draw the label text on the image
        cv2.putText(img, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self):
        self.img = cv2.imread(self.input_image) if  not isinstance(self.input_image, np.ndarray) else self.input_image
        self.img_height, self.img_width = self.img.shape[:2]
        letterbox = LetterBox(new_shape=self.size, auto=False, stride=32)
        image = np.stack([letterbox(image=self.img)])
        image = (image[..., ::-1]).transpose((0, 3, 1, 2))
        image = np.ascontiguousarray(image).astype(np.float32)
        return image / 255

    def postprocess(self, input_image, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        pred = np.transpose(output[0]) 
        x = pred[:, 0] - pred[:, 2] / 2
        y = pred[:, 1] - pred[:, 3] / 2
        w = pred[:, 2]
        h = pred[:, 3]
        boxes = np.vstack([x, y, w, h]).T
        class_ids = np.argmax(pred[:, 4:], axis=1)
        scores = np.max(pred[:, 4:], axis=1)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            gain = min(self.size[1] / self.img_width, self.size[0] / self.img_height)
            pad = (
                round((self.size[1] - self.img_width * gain) / 2 - 0.1),
                round((self.size[0] - self.img_height * gain) / 2 - 0.1),
            )
            box[0] = (box[0] - pad[0]) / gain
            box[1] = (box[1] - pad[1]) / gain
            box[2] = box[2] / gain
            box[3] = box[3] / gain
            score = scores[i]
            class_id = class_ids[i]

            if score > 0.25:
                # Draw the detection on the input image
                self.draw_detections(input_image, box, score, class_id)
        return input_image

    def main(self, input_image):
        """
        Performs inference using a TFLite model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """
        self.input_image = input_image 
        # Create an interpreter for the TFLite model

        interpreter = self.model
        # Get the model inputs
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Store the shape of the input for later use
        input_shape = input_details[0]["shape"]
        self.input_width = input_shape[1]
        self.input_height = input_shape[2]

        # Preprocess the image data
        img_data = self.preprocess()
        self.img_data = img_data
        # img_data = img_data.cpu().numpy()
        # Set the input tensor to the interpreter
        img_data = img_data.transpose((0, 2, 3, 1))

        scale, zero_point = input_details[0]["quantization"]
        interpreter.set_tensor(input_details[0]["index"], img_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        output[:, [0, 2]] *= self.size[1]
        output[:, [1, 3]] *= self.size[0]
        return self.postprocess(self.img, output)
