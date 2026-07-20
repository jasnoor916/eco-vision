from ultralytics import YOLO

def main():
    data = r"C:\Users\Jun Ci\workspace\gretchen\gretchen_project\eco-vision\data\data.yaml"
    model_cfg = "yolo26-p2.yaml"

    model = YOLO(model_cfg)

    model.train(
        data=data,
        epochs=100,
        imgsz=640,
        device="cuda",
    )

if __name__ == "__main__":
    main()