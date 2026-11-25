import cv2
import numpy as np
import os
import urllib.request
import ssl


def download_cascade(url, filename):
    """Downloads Haar cascade XML files if they are missing locally."""
    if not os.path.exists(filename):
        print(f"Downloading missing cascade file: {filename}...")
        try:
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(url, context=context) as response, open(filename, 'wb') as out_file:
                out_file.write(response.read())
            print(f"Successfully downloaded {filename}.")
        except Exception as e:
            print(f"Failed to download {filename}. Error: {e}")


def process_image_for_anonymization(image_path, output_filename):
    if not os.path.exists(image_path):
        print(f"Error: '{image_path}' not found.")
        return

    # 1. Import the original image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Pre-processing: CLAHE for illumination correction
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    # --- OPTIMIZATION 1: Image Pyramid Expansion ---
    # Magnify the grayscale image by 2x. This physically enlarges the tiny
    # background Inception faces so they exceed the Haar cascade's minimum training size.
    scale_multiplier = 2.0
    gray_large = cv2.resize(gray_clahe, (0, 0), fx=scale_multiplier, fy=scale_multiplier)

    # 3. Load Haar Cascade Classifiers
    # --- OPTIMIZATION 2: Switch to 'alt2' model ---
    # The alt2 model is significantly better at handling slightly rotated faces
    # (like the characters in Inception) and is less easily tricked by R2-D2's panels.
    face_cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
    eye_cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"

    download_cascade(face_cascade_url, "haarcascade_frontalface_alt2.xml")
    download_cascade(eye_cascade_url, "haarcascade_eye.xml")

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

    # 4. Detect Faces on the ENLARGED image
    faces_large = face_cascade.detectMultiScale(
        gray_large,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(30, 30)  # 30x30 on the large image is effectively 15x15 on the original
    )

    valid_faces_count = 0

    # 5. Extract, Validate, and Blur
    for (lx, ly, lw, lh) in faces_large:
        # Map the magnified coordinates back to the original image dimensions
        x = int(lx / scale_multiplier)
        y = int(ly / scale_multiplier)
        w = int(lw / scale_multiplier)
        h = int(lh / scale_multiplier)

        # Extract regions from the original-sized images
        roi_gray = gray_clahe[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # Restrict eye search to the upper 65% of the candidate face
        roi_gray_upper = roi_gray[0:int(h * 0.65), :]

        # --- OPTIMIZATION 3: Enlarged ROI Eye Detection ---
        # Distant eyes are incredibly small. Upscale the eye search region by 2x
        # so the eye cascade can actually see the pixel gradients of the irises.
        roi_upper_large = cv2.resize(roi_gray_upper, (0, 0), fx=2.0, fy=2.0)

        eyes_large = eye_cascade.detectMultiScale(
            roi_upper_large,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(10, 10)
        )

        # Feature Validation Heuristic: Ensure eyes exist to filter out R2-D2
        if len(eyes_large) == 0:
            continue

        # If validated, draw the bounding box on the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        valid_faces_count += 1

        # Apply Gaussian Blur to the verified eyes
        for (ex_L, ey_L, ew_L, eh_L) in eyes_large:
            # Map magnified eye coordinates back to original face ROI scale
            ex = int(ex_L / 2.0)
            ey = int(ey_L / 2.0)
            ew = int(ew_L / 2.0)
            eh = int(eh_L / 2.0)

            eye_region = roi_color[ey:ey + eh, ex:ex + ew]
            blurred_eye = cv2.GaussianBlur(eye_region, (23, 23), 30)
            roi_color[ey:ey + eh, ex:ex + ew] = blurred_eye

    # 6. Save the processed image
    cv2.imwrite(output_filename, img)
    print(f"Success! Processed and saved: {output_filename} (Valid Human Faces confirmed: {valid_faces_count})")


def main():
    print("Starting Upscaled Data Anonymization Pipeline...\n")

    targets = [
        ("inception.jpg", "anonymized_inception.jpg"),
        ("obi-wan-kenobi.jpg", "anonymized_obi-wan.jpg"),
        ("r2d2.jpg", "anonymized_r2d2.jpg")
    ]

    for input_img, output_img in targets:
        process_image_for_anonymization(input_img, output_img)

    print("\nPipeline complete. Distant subjects are now dynamically scaled and detected.")


if __name__ == "__main__":
    main()