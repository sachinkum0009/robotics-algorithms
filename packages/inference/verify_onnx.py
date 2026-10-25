import onnxruntime as ort
ort.get_available_providers()

providers = ort.get_available_providers()
print("Available ONNX Runtime providers:")
for provider in providers:
    print(f"- {provider}")