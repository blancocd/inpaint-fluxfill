from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils.loading_utils import load_image

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
)
pipe.to("cuda")
prompt = "Front view and back view of the same person wearing a marathon athletic shirt which has sponsors printed on it"
#image and mask_image should be PIL images.
#The mask structure is white for inpainting and black for keeping as is
root = "/mnt/lustre/work/ponsmoll/pba534/inpaint/examples/62/"
image = load_image(root+"image.png")
mask_image = load_image(root+"mask1.png")
image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
image.save(root+"test.png")

