from main import *
import cv2
import time

model_path = "raf-db-model_best.pth"

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model = pyramid_trans_expr2(img_size=224, num_classes=7)

model = torch.nn.DataParallel(model)
model = model.to(device)
currtime = time.strftime("%H:%M:%S")
print(currtime)


def main():
    if model_path is not None:
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path, map_location=device)
            best_acc = checkpoint["best_acc"]
            best_acc = best_acc.to()
            print(f"best_acc:{best_acc}")
            model.load_state_dict(checkpoint["state_dict"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    model_path, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(model_path))
        imagecapture(model)
        return


def imagecapture(model):
    currtimeimg = time.strftime("%H:%M:%S")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    start_time = None
    capturing = False

    while True:
        from prediction import predict

        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
        )

        # Display the frame
        cv2.imshow("Webcam", frame)

        # If faces are detected, start the timer
        if len(faces) > 0:
            print(f"[!]Face detected at {currtimeimg}")
            face_region = frame[
                faces[0][1] : faces[0][1] + faces[0][3],
                faces[0][0] : faces[0][0] + faces[0][2],
            ]  # Crop the face region
            face_pil_image = Image.fromarray(
                cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            )  # Convert to PIL image
            print("[!]Start Expressions")
            print(f"-->Prediction starting at {currtimeimg}")
            predictions = predict(model, image_path=face_pil_image)
            print(f"-->Done prediction at {currtimeimg}")

            # Reset capturing
            capturing = False

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the webcam and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
