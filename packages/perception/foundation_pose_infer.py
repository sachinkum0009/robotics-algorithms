import cv2
import numpy as np
import onnxruntime as ort


class FoundationPoseInference:
    def __init__(self, refine_path: str, score_path: str):
        # Initialize sessions
        self.refine_sess = ort.InferenceSession(
            refine_path, providers=["CUDAExecutionProvider"]
        )
        self.score_sess = ort.InferenceSession(
            score_path, providers=["CUDAExecutionProvider"]
        )

    def preprocess(self, rgb: cv2.Mat, depth: cv2.Mat):
        """
        Normalize and prepare images.
        FoundationPose expects: B x 3 x 160 x 160
        """
        # 1. Resize/Crop to 160x160 (Assumes you've already cropped to the object bbox)
        img = cv2.resize(rgb, (160, 160))
        img = img.astype(np.float32) / 255.0
        # Standard ImageNet Normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)  # HWC to CHW
        return np.expand_dims(img, axis=0)

    def run_inference(self, obs_rgb, obs_depth, rendered_rgb, rendered_depth):
        # Preprocess both observed and rendered pairs
        input_obs = self.preprocess(obs_rgb, obs_depth)
        input_rendered = self.preprocess(rendered_rgb, rendered_depth)

        # 1. Refinement Step
        # The model usually takes [batch, 6, H, W] (RGB-RGB or RGBD-RGBD concatenated)
        # OR separate inputs depending on your specific export.
        # Check your input names with: self.refine_sess.get_inputs()
        refine_inputs = {
            "input_query": input_obs,
            "input_reference": input_rendered,
        }
        refine_outputs = self.refine_sess.run(None, refine_inputs)

        # 2. Scoring Step
        # Takes the output of refine to rank the best pose
        score_inputs = {"input": refine_outputs[0]}
        confidence_scores = self.score_sess.run(None, score_inputs)

        return refine_outputs, confidence_scores


# Usage
fps = FoundationPoseInference("refine_model.onnx", "score_model.onnx")

# Note: You MUST provide 'rendered_rgb' by rendering your CAD model
# from an initial estimated pose (e.g., from a detector like CosyPose or YOLO).
# results = fps.run_inference(real_crop, real_depth, render_crop, render_depth)
