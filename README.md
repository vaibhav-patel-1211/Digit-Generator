# Digit-Generator

A Flask web application that generates handwritten digits using a Conditional Generative Adversarial Network (cGAN). Users can select any digit from 0-9 and the application will generate a 5x5 grid of synthetic handwritten digit images.

## Features

- Generate handwritten digits (0-9) using a trained cGAN model
- Interactive web interface
- Real-time digit generation
- 5x5 grid display of generated samples

## Prerequisites

Before running this project, make sure you have the following installed:

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/vaibhav-patel-1211/Digit-Generator.git
   cd Digit-Generator
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   ```

   **On Windows:**

   ```bash
   venv\Scripts\activate
   ```

   **On macOS/Linux:**

   ```bash
   source venv/bin/activate
   ```

3. **Install required dependencies**
   ```bash
   pip install flask torch torchvision
   ```

## How to Run

1. **Make sure you have the model files**

   - Ensure that `generator_cgan.pth` is present in the project root directory
   - The application will automatically load this model when it starts

2. **Start the Flask application**

   ```bash
   python app.py
   ```

3. **Access the web interface**

   - Open your web browser and navigate to: `http://localhost:5000`
   - You should see the digit generator interface

4. **Generate digits**
   - Enter a digit between 0 and 9 in the input field
   - Click the "Generate" button
   - A 5x5 grid of generated handwritten digits will be displayed

## Project Structure

```
Digit-Generator/
│
├── app.py                      # Flask application and model code
├── generator_cgan.pth          # Trained generator model weights
├── discriminator_cgan.pth      # Trained discriminator model weights (not used in inference)
├── templates/
│   └── index.html             # Web interface template
└── README.md                   # This file
```

## Usage

1. Launch the application by running `python app.py`
2. In the web browser, enter any digit from 0 to 9
3. Click "Generate" to create a grid of 25 synthetic digit images
4. The generated images will appear below the form

## Technical Details

- **Framework**: Flask
- **Deep Learning Library**: PyTorch
- **Model Type**: Conditional GAN (cGAN)
- **Latent Dimension**: 100
- **Image Size**: 28x28 pixels (MNIST-style)
- **Number of Classes**: 10 (digits 0-9)
- **Output**: 5x5 grid (25 samples per generation)

## Notes

- The application automatically uses GPU if available (CUDA), otherwise it falls back to CPU
- Model weights are loaded at startup, so the first generation may take slightly longer
- The application runs in debug mode by default for development purposes

## Troubleshooting

- **Model file not found**: Ensure `generator_cgan.pth` is in the project root directory
- **Port already in use**: Change the port number in `app.py` (line 128) if port 5000 is occupied
- **Import errors**: Make sure all dependencies are installed correctly using `pip install flask torch torchvision`

## License

This project is open source and available for educational purposes.
