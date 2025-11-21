"""
Flask frontend for the conditional GAN digit generator.

This application loads the trained generator weights and exposes
an HTML form where a user can choose a digit between 0 and 9.
When the form is submitted, the generator produces a 5x5 grid
of images for the requested digit, renders it to an in-memory
PNG, and returns it to the browser as an inline image.
"""

import base64
import io
from typing import Optional

from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import functional as TF


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """Generator architecture matching the training script."""

    def __init__(self, latent_dim: int, num_classes: int, image_size: int) -> None:
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, image_size),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass that concatenates noise and label embedding before decoding."""
        gen_input = torch.cat((noise, self.label_emb(labels)), dim=-1)
        return self.model(gen_input)


# ---------------------------------------------------------------------------
# Flask setup
# ---------------------------------------------------------------------------

app = Flask(__name__)


# Global configuration constants used by the generator.
LATENT_DIM = 100
NUM_CLASSES = 10
IMAGE_SIZE = 28 * 28
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the generator and load trained weights at startup.
generator = Generator(LATENT_DIM, NUM_CLASSES, IMAGE_SIZE).to(DEVICE)
generator.load_state_dict(torch.load("generator_cgan.pth", map_location=DEVICE))
generator.eval()


def generate_digit_grid(digit: int, samples: int = 25) -> Optional[str]:
    """
    Generate a base64-encoded PNG grid for the requested digit.

    Returns a string suitable for embedding in an <img> src attribute.
    """
    if digit < 0 or digit >= NUM_CLASSES:
        return None

    with torch.no_grad():
        noise = torch.randn(samples, LATENT_DIM, device=DEVICE)
        labels = torch.full((samples,), digit, dtype=torch.long, device=DEVICE)
        generated = generator(noise, labels).view(-1, 1, 28, 28)

    # Convert model outputs from [-1, 1] back to [0, 1] for visualization.
    generated = (generated + 1) / 2

    grid = torchvision.utils.make_grid(generated, nrow=5)

    # Serialize the grid image to PNG in memory.
    buffer = io.BytesIO()
    TF.to_pil_image(grid).save(buffer, format="PNG")
    buffer.seek(0)

    # Encode the PNG bytes to base64 for embedding in the HTML template.
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Render the home page where users can request digit generations.

    On POST, generate a grid for the chosen digit and pass it to the template.
    """
    generated_image = None
    selected_digit = None
    error_message = None

    if request.method == "POST":
        digit_str = request.form.get("digit", "").strip()
        if digit_str.isdigit():
            selected_digit = int(digit_str)
            generated_image = generate_digit_grid(selected_digit)
            if generated_image is None:
                error_message = "Please enter a number between 0 and 9."
        else:
            error_message = "Please enter a valid integer digit."

    return render_template(
        "index.html",
        generated_image=generated_image,
        selected_digit=selected_digit,
        error_message=error_message,
    )


if __name__ == "__main__":
    # Enable Flask's debug mode for easier development and auto-reloads.
    app.run(host="0.0.0.0", port=5000, debug=True)

