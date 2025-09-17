from ultralytics import YOLO

def main():
    # Load your trained model
    model = YOLO("best.pt")

    # Run prediction on an image
    results = model.predict(source="test.jpg", show=True, save=True)

    # Run prediction on a video
    # results = model.predict(source="video.mp4", show=True, save=True)

    # Run prediction using webcam (0 = default camera)
    # results = model.predict(source=0, show=True)

    print("✅ Inference completed. Results are saved in 'runs/detect/predict/'")

if __name__ == "__main__":
    main()
