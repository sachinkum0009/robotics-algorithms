import numpy as np
import onnx
import onnxruntime as ort
from matplotlib import patches
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms

provider_options = [
    {
        "ModelFormat": "MLProgram",
        "ComputeUnits": "ALL",  # Use CPU, GPU, and Neural Engine
    }
]
# 1. Setup the Session
# providers = ["CPUExecutionProvider"]
model_path = "/Users/mac/zzzzz/python/robotics-algorithms/packages/perception\
            /yolo_model_static.onnx"
test_model = onnx.load(model_path)
print(onnx.checker.check_model(test_model))
print([n.op_type for n in test_model.graph.node])
providers = [
    (
        "CoreMLExecutionProvider",
        {
            "ModelFormat": "MLProgram",
            "MLComputeUnits": "ALL",
            "RequireStaticInputShapes": "0",
            "EnableOnSubgraphs": "0",
        },
    ),
]
session = ort.InferenceSession(model_path, providers=providers)
# VOC Classes for labeling
VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    transform = transforms.Compose(
        [
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ]
    )
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor.numpy(), img, original_size


def run_ort_inference(image_path, threshold=0.5):
    input_data, original_img, (w_orig, h_orig) = preprocess(image_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})

    predictions = outputs[0]  # Shape: [1, 7, 7, 75]
    predictions = predictions.reshape(7, 7, 3, 25)

    detections = []
    S = 7  # Grid size

    for i in range(S):
        for j in range(S):
            for a in range(3):
                data = predictions[i, j, a]
                # Apply sigmoid to confidence
                conf = 1 / (1 + np.exp(-data[4]))

                if conf > threshold:
                    # 1. Get Class with highest probability
                    class_probs = data[5:]
                    # Softmax for cleaner class scores
                    class_probs = np.exp(class_probs) / np.sum(
                        np.exp(class_probs)
                    )
                    class_id = np.argmax(class_probs)

                    # 2. Convert grid-relative coords to image-relative (0-1)
                    # data[0:2] are x, y relative to the cell [0, 1]
                    # data[2:4] are w, h relative to the whole image
                    x_cell, y_cell, w_img, h_img = data[0:4]

                    x_relative = (j + x_cell) / S
                    y_relative = (i + y_cell) / S

                    detections.append(
                        {
                            "box": [x_relative, y_relative, w_img, h_img],
                            "conf": conf,
                            "class": VOC_CLASSES[class_id],
                        }
                    )

    return detections, original_img


def plot_results(img: Image.Image, detections: list):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    w_orig, h_orig = img.size

    for det in detections:
        x_c, y_c, w, h = det["box"]

        # Convert 0-1 relative coordinates to absolute pixel coordinates
        # Center-coords to top-left corner
        width = w * w_orig
        height = h * h_orig
        xmin = (x_c * w_orig) - (width / 2)
        ymin = (y_c * h_orig) - (height / 2)

        rect = patches.Rectangle(
            (xmin, ymin),
            width,
            height,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        label = f"{det['class']} {det['conf']:.2f}"
        plt.text(
            xmin,
            ymin,
            label,
            color="white",
            verticalalignment="bottom",
            bbox={"color": "red", "pad": 0},
        )

    plt.show()


# Usage with a custom threshold
img_path = (
    "/Users/mac/zzzzz/python/robotics-algorithms/packages/perception/data/\
    VOCdevkit/VOC2012/JPEGImages/2012_004147.jpg"
)
results, img = run_ort_inference(
    img_path, threshold=0.52
)  # Set your threshold here
plot_results(img, results)
