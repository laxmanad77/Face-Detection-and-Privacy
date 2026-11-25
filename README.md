# Face Detection and Privacy

A Python application that detects human faces in images and automatically blurs the eyes to improve privacy.

## Features

- Detects human faces from an input image
- Detects eye regions within faces
- Applies a blur effect to protect eye privacy
- Saves the processed image
- Uses OpenCV for image processing

## Requirements

- Python 3.8 or newer
- OpenCV

## Installation

1. Clone the repository:

```bash
git clone https://github.com/laxmanad77/Face-Detection-and-Privacy.git
cd Face-Detection-and-Privacy
```

2. Install the required dependency:

```bash
pip install opencv-python
```

## Usage

Place the image you want to process in the project directory, then run:

```bash
python main.py
```

Update the image path and output settings in the script if necessary.

## How It Works

1. Loads an image using OpenCV.
2. Converts the image to grayscale.
3. Uses Haar cascade classifiers to detect faces and eyes.
4. Applies a Gaussian blur to the detected eye regions.
5. Displays or saves the resulting image.

## Privacy Notice

This project is intended to help reduce identifying facial details in images. It does not guarantee complete anonymization. Lighting, image quality, facial orientation, and model limitations may affect detection accuracy.

## Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a pull request.

## License

No license has been specified for this repository.

## Author

[laxmanad77](https://github.com/laxmanad77)
