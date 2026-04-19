from PIL import Image

def convert_image(image, target_format):
    target_format = target_format.upper()
    
    # Formats that do not support Alpha channel (transparency)
    no_alpha_formats = ["JPEG", "BMP", "ICO"]
    
    if target_format in no_alpha_formats:
        # Check if image has alpha, if so, convert to RGB (white background)
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3] if image.mode == 'RGBA' else None)
            return background
        else:
            return image.convert("RGB")
            
    return image
