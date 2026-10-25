import onnxruntime as ort


class YoloInference:
    def __init__(self):
        self._model = ort.SessionOptions()

    def run_inference(self, input):
        pass
