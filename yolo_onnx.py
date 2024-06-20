import cv2
import numpy as np
import onnxruntime as ort


class YoloOnnx:
    def __init__(self, model, size, confidence_thres, iou_thres, num_threads, **kwargs):
        self.size = size
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        session = ort.InferenceSession(model, sess_options, providers=["CPUExecutionProvider"])
        self.model = session
        
        self.model_inputs = session.get_inputs()
        input_shape = self.model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]


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
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Read the input image using OpenCV
        self.img = cv2.imread(self.input_image) if  not isinstance(self.input_image, np.ndarray) else self.input_image

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data


    def postprocess(self, input_image, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        pred = np.transpose(np.squeeze(output[0]))

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height
        
        x = (pred[:, 0] - pred[:, 2] / 2) * x_factor
        y = (pred[:, 1] - pred[:, 3] / 2) * y_factor
        w = pred[:, 2] * x_factor
        h = pred[:, 3] * y_factor
        boxes = np.vstack([x, y, w, h]).T
        class_ids = np.argmax(pred[:, 4:], axis=1)
        scores = np.max(pred[:, 4:], axis=1)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

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

        # Preprocess the image data
        img_data = self.preprocess()

        # Run inference using the preprocessed image data
        outputs = self.model.run(None, {self.model_inputs[0].name: img_data})

        # Perform post-processing on the outputs to obtain output image.
        return self.postprocess(self.img, outputs)  # output image

        
