import torch
from tieUtils.get_mask import Image_mask
from tieUtils.load_models import load_masker_model, load_MMOCR
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter

def enhance(image):
    contrast_enhancer = ImageEnhance.Contrast(image)
    enhanced_image = contrast_enhancer.enhance(4.0)
    return enhanced_image

def erode(cycles, image, pixel):
    for _ in range(cycles):
         image = image.filter(ImageFilter.MinFilter(pixel))
    return image


def dilate(cycles, image, pixel):
    for _ in range(cycles):
         image = image.filter(ImageFilter.MaxFilter(pixel))
    return image

def predict_textboxes(img):
    mmocr_inferencer = load_MMOCR()
    result = mmocr_inferencer(img)['predictions'][0]
    rec_texts = result['rec_texts']
    det_polygons = result['det_polygons']
    mask_getter = Image_mask('1.jpg','1.jpg',det_polygons)
    mask = mask_getter.get_mask('maske.png')
    return mask

def predict_mask(image_path, enhance_contrast=False):
    model, device = load_masker_model()
    # Define the image transform
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path).convert('RGB')
    if enhance_contrast:
        img = enhance(img)
    img = transform(img).unsqueeze(0)

    # Move the image to the device
    img = img.to(device)

    # Make a prediction
    with torch.no_grad():
        output = model(img)
        # Apply sigmoid function to normalize the output to [0, 1]
        output = torch.sigmoid(output)

    # Convert the output to a PIL image and save it
    output = output.cpu().squeeze().numpy()
    output = (output > 0.1).astype('uint8') * 255
    # output = (output > 0.01).astype('uint8') * 255
    output = Image.fromarray(output, mode='L')

    dilated_image = dilate(2,output,5)

    output.save('out.png')
    return dilated_image