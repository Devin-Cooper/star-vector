from PIL import Image
from starvector.model.starvector_arch import StarVectorForCausalLM
from starvector.data.util import process_and_rasterize_svg
import torch
import gc

# Force garbage collection before loading model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    torch.mps.empty_cache()

model_name = "starvector/starvector-1b-im2svg"
# model_name = "starvector/starvector-8b-im2svg"

# On Mac, disable flash attention - it's CUDA-only
starvector = StarVectorForCausalLM.from_pretrained(
    model_name,
    use_flash_attn=False,  # Explicitly disable flash attention for Mac compatibility
    use_cache=False,      # Disable KV caching to save memory
    torch_dtype=torch.float16  # Use float16 for better performance
)

# Set the tokenizer's model max length directly
if hasattr(starvector.model, 'svg_transformer') and hasattr(starvector.model.svg_transformer, 'tokenizer'):
    starvector.model.svg_transformer.tokenizer.model_max_length = 2048
    print(f"Set tokenizer model_max_length to 2048")

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

# Clear memory before inference
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    torch.mps.empty_cache()

image_pil = Image.open('assets/examples/sample-0.png')
image = starvector.process_images([image_pil])[0].to(torch.float16).to(device)
batch = {"image": image}

# Use more conservative generation parameters for Mac Studio
raw_svg = starvector.generate_im2svg(
    batch, 
    max_new_tokens=1000,  # Increase token limit for better output
    max_length=2048,      # Set explicit max_length to match tokenizer
    temperature=0.7,      # Higher temperature for more creative outputs
    repetition_penalty=1.5,  # Reduced penalty
    do_sample=True,
    num_beams=4,          # Increase beam search
    use_cache=True        # Try with cache enabled for better quality
)[0]

# Print debug information about the generated SVG
print(f"SVG length: {len(raw_svg)}")

# Clear GPU memory immediately after generation
del image, batch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    torch.mps.empty_cache()

svg, raster_image = process_and_rasterize_svg(raw_svg)
print(svg)
