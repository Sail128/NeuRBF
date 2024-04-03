from PIL import Image

image_path = "pluto.png"
image = Image.open(image_path)

sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000]

for size in sizes:
    resized_image = image.resize((size, size))
    resized_image.save(f"pluto_{size}.png")