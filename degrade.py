from PIL import Image

# Open original high-res image
img = Image.open('data/test/0/IMG_1199.JPG').convert('RGB')

# Simulate low quality by resizing down and then back up
low_res = img.resize((32, 32), Image.BICUBIC)  # degrade
degraded_img = low_res.resize((128, 128), Image.BICUBIC)  # upscale

# Save and use this degraded image
degraded_img.save('degraded_test.jpg')
