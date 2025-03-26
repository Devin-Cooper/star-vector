from PIL import Image
from starvector.model.starvector_arch import StarVectorForCausalLM
from starvector.data.util import process_and_rasterize_svg
import torch

model_name = "starvector/starvector-1b-im2svg"
# model_name = "starvector/starvector-8b-im2svg"

starvector = StarVectorForCausalLM.from_pretrained(model_name)

# Choose appropriate device
if torch.cuda.is_available():
    device = "cuda"
    starvector.cuda()
elif torch.backends.mps.is_available():
    device = "mps"
    starvector.to(device)
else:
    device = "cpu"
    print("WARNING: Using CPU for inference which will be very slow")
    
starvector.eval()

image_pil = Image.open('assets/examples/sample-0.png')
image = starvector.process_images([image_pil])[0].to(torch.float16).to(device)
batch = {"image": image}

raw_svg = starvector.generate_im2svg(batch, max_length=4000, temperature=1.5, length_penalty=-1, repetition_penalty=3.1)[0]
svg, raster_image = process_and_rasterize_svg(raw_svg)

print(svg)
