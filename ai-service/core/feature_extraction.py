import cv2
import numpy as np
import onnxruntime as ort
import albumentations as A


class FeatureExtraction:
    def __init__(
        self,
        onnx_path="./models/pretrained_models/osnet_x1_0.onnx",
        device="cpu",
    ):
        self.onnx_path = onnx_path
        self.device = device
        if self.device.lower() == "cpu":
            if ort.get_device() == "CUDA":
                print("CUDA available, if you want to switch your device into CUDA")
            self.ort_session = ort.InferenceSession(
                self.onnx_path, providers=["CPUExecutionProvider"]
            )
        elif self.device.lower() == "cuda":
            self.ort_session = ort.InferenceSession(
                self.onnx_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        else:
            raise ValueError("Choose between CPU or CUDA!")
        self.model_height, self.model_width = self.ort_session.get_inputs()[0].shape[
            2:4
        ]
        self.image_augmentation = A.Compose(
            [
                A.Resize(self.model_height, self.model_width),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def predict_img(self, img):
        image = self._preprocessing_img(img)
        input_onnx = self.ort_session.get_inputs()[0].name
        output_onnx = self.ort_session.run(None, {input_onnx: image})
        return output_onnx

    def _preprocessing_img(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = self.image_augmentation(image=np.array(image))["image"]
        # Convert to CHW format and add batch dimension
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = np.expand_dims(image, axis=0)
        return image
