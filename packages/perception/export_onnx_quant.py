from onnxruntime.quantization import QuantType, quantize_dynamic

quantize_dynamic(
    "yolo_model.onnx", "yolo_model_int8.onnx", weight_type=QuantType.QUInt8
)
