from ultralytics import RTDETR

model = RTDETR("RT_DETR_adult.pt")

model.export(
    format="onnx",
    imgsz=(544,960),
    opset=16,
    simplify=True
)