import torch
from models.yolo import YOLO

# from utils import DEVICE
DEVICE = "cpu"

model = YOLO(7, 20, 3)
state_dict = torch.load("yolo_model_100.pth", map_location=torch.device("cpu"))
model.load_state_dict(state_dict)
model.eval()
dummy_input = torch.randn(1, 3, 448, 448).to(DEVICE)
onnx_path = "yolo_model_100_static.onnx"


torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,  # Store trained weights in the file
    opset_version=18,  # High compatibility version
    do_constant_folding=True,  # Optimization
    input_names=["input"],
    output_names=["output"],
    # dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    dynamic_axes=None,
)

print(f"Model exported to {onnx_path}")
