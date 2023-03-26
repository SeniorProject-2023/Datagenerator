import random

from PIL import Image

from probe_relighting.utils.demotools import *
from probe_relighting.utils.preprocessing import image_tr

args = {
    "mode": "synthetic",
    "step": 20,
    "intensity": 25
}

# Loading model with weights
model = get_model()


# read images from original folder
# img_list = get_images()

def generate_outputs(model, img, random_seed, style):
    model.to(device)
    minr = 1
    maxr = 361
    with torch.no_grad():
        model.eval()
        idx = random.Random(random_seed).randrange(minr, maxr)
        sample = make_sample(img, style, idx)
        output = model(sample)
        output = output['generated_img']
        output = denorm(output.cpu().squeeze())
        output = to_pil_image(output)
        return output


def generate_relighted_image(img, random_seed):
    size = img.size
    image = img.convert("RGB")
    image = image_tr(image)
    image = image.to(device)
    img_input = image.unsqueeze(0)
    output = generate_outputs(model, img_input, random_seed, style=args["mode"])
    output = output.convert("L")
    output = output.resize(size)
    return output


if __name__ == "__main__":
    save_path = Path(f'./output/')
    save_path.mkdir(parents=True, exist_ok=True)
    name = Path(save_path, '0.jpg')
    img = Image.open("/home/ramez/PycharmProjects/Datagenerator/output/0.png")
    generate_relighted_image(img, random.random()).save(name)
