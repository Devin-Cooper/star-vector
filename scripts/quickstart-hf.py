from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from starvector.data.util import process_and_rasterize_svg
import torch

# model_name = "starvector/starvector-1b-im2svg"
model_name = "starvector/starvector-8b-im2svg"

torch_dtype = torch.float16

# Choose appropriate device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
    print("WARNING: Using CPU for inference which will be very slow")

starvector = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, trust_remote_code=True)
processor = starvector.model.processor
tokenizer = starvector.model.svg_transformer.tokenizer

starvector.to(device)
starvector.eval()

image_pil = Image.open('assets/examples/sample-18.png')

image = processor(image_pil, return_tensors="pt")['pixel_values'].to(device)
if not image.shape[0] == 1:
    image = image.squeeze(0)
batch = {"image": image}

raw_svg = starvector.generate_im2svg(batch, max_length=100)[0]
svg, raster_image = process_and_rasterize_svg(raw_svg)

print(svg)
