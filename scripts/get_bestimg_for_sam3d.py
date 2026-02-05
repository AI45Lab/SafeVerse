import os
import base64
import json
import mimetypes
import shutil
import io
from PIL import Image
from openai import OpenAI
import argparse

API_KEY = "xxx"
ENDPOINT = "xxx" 
DEPLOYMENT_NAME = "gpt-4o"      
API_VERSION = "2024-10-21" 

def resize_image_for_api(image, max_size=768):

    width, height = image.size
    if max(width, height) > max_size:
        ratio = max_size / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def get_best_image_for_sam3d(folder_path, folder_name, client):
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_exts and f.endswith('_orig.png')]
    
    if not images:
        return None, None
    images.sort()
    max_images_to_analyze = 10
    selected_candidates = images[:max_images_to_analyze]

    content_list = []
    
    prompt_text = f"""
    You are an expert **3D Reconstruction Specialist**.
    Select the SINGLE BEST image of the object "{folder_name}" for 3D reconstruction (SAM-3D/Trellis).
    
    **Criteria:**
    1. Maximum visible surface area (Canonical view).
    2. Complete object visibility (No cropping).
    3. Clean geometry.

    **Output JSON:**
    {{ "best_image": "filename_orig.png", "reason": "brief reason" }}
    """
    content_list.append({"type": "text", "text": prompt_text})

    valid_imgs_count = 0

    for img_file in selected_candidates:
        img_path = os.path.join(folder_path, img_file)
        mask_file = img_file.replace('_orig.png', '_mask.png')
        mask_path = os.path.join(folder_path, mask_file)

        original_img = Image.open(img_path).convert("RGBA")
        
        final_img = original_img
        if os.path.exists(mask_path):
            mask_img = Image.open(mask_path).convert("L")
            if mask_img.size != original_img.size:
                mask_img = mask_img.resize(original_img.size)
            
            white_bg = Image.new("RGBA", original_img.size, (255, 255, 255, 255))
            final_img = Image.composite(original_img, white_bg, mask_img)

        final_img = final_img.convert("RGB")
        final_img = resize_image_for_api(final_img, max_size=768)

        buffered = io.BytesIO()
        final_img.save(buffered, format="JPEG", quality=85)
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        content_list.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "low" 
            }
        })
        content_list.append({"type": "text", "text": f"Image Filename: {img_file}"})
        valid_imgs_count += 1

    if valid_imgs_count == 0:
        return None, None

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are a 3D reconstruction expert."},
            {"role": "user", "content": content_list}
        ],
        max_tokens=300,
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    
    result_json = json.loads(response.choices[0].message.content)
    best_filename = result_json.get("best_image")
    reason = result_json.get("reason")
    
    return best_filename, reason


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instances_images', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()
    return args 

def main():
    args = options()
    input_instance_dir = args.instances_images
    out_dir = args.out_dir

    client = OpenAI(
        api_key=API_KEY,
        base_url=f"{ENDPOINT}openai/v1/",
        default_headers={"Ocp-Apim-Subscription-Key": API_KEY} 
    )

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    for root, dirs, files in os.walk(input_instance_dir):
        if root == input_instance_dir:
            continue

        folder_name = os.path.basename(root)
    
        best_file, reason = get_best_image_for_sam3d(root, folder_name, client)
        
        if not best_file:
            valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
            fallback_images = [f for f in os.listdir(root) if os.path.splitext(f)[1].lower() in valid_exts and f.endswith('_orig.png')]
            
            if fallback_images:
                fallback_images.sort() 
                best_file = fallback_images[0]

        if best_file:
            full_source_path = os.path.join(root, best_file)
            file_ext = os.path.splitext(best_file)[1]
            
            destination_path = os.path.join(out_dir, f"{folder_name}_ori{file_ext}")

            if 'orig' in best_file:
                mask_source_name = best_file.replace('orig', 'mask')
            else:
                mask_source_name = best_file.replace('_orig', '_mask') 
                
            p1_mask = os.path.join(root, mask_source_name)
            p2_mask = os.path.join(out_dir, f"{folder_name}_mask{file_ext}")

            if os.path.exists(p1_mask):
                shutil.copy(p1_mask, p2_mask)
            else:
                alt_mask_name = best_file.split('_')[0] + "_mask" + file_ext 
                p1_alt = os.path.join(root, alt_mask_name)
                if os.path.exists(p1_alt):
                    shutil.copy(p1_alt, p2_mask)

            shutil.copy(full_source_path, destination_path)

if __name__ == "__main__":
    main()
    print("=== done", flush=True)