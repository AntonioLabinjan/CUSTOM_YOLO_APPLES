from ultralytics import YOLO

model = YOLO("yolov11_custom.pt")

model.predict(source = "Image_6.jpeg", show=True, save=True, conf = 0.6, save_crop = True, save_txt = True)
