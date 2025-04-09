import os, pdb
import argparse
import numpy as np
import torch
import requests
import lpips
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
from transformers import CLIPProcessor, CLIPModel
from skimage.metrics import structural_similarity as ssim
import cv2

from diffusers import DEISScheduler
from utils.edit_directions import construct_direction
from utils.edit_pipeline import EditingPipeline

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def compute_clip_score(image, text, model, processor):
    """计算CLIP score"""
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    return outputs.logits_per_image.item()

def compute_bg_lpips(img1, img2, lpips_model, mask=None):
    """
    Compute LPIPS distance focusing on background regions
    
    Args:
        img1, img2: PIL Images to compare
        lpips_model: LPIPS model
        mask: Foreground mask (1 for foreground, 0 for background). If None, uses full image.
    
    Returns:
        LPIPS distance score focusing on the background regions
    """
    # Convert PIL images to tensors in range [-1, 1] as expected by LPIPS
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)
    ])
    
    tensor1 = transform(img1).unsqueeze(0).to(device)
    tensor2 = transform(img2).unsqueeze(0).to(device)
    
    if mask is None:
        # No mask provided, compute LPIPS on entire image
        with torch.no_grad():
            distance = lpips_model(tensor1, tensor2)
        return distance.item()
    
    # Convert mask to tensor and ensure it has the same size as input images
    mask_tensor = to_tensor(mask).unsqueeze(0).to(device)
    
    # Invert mask to focus on background (1 - mask)
    bg_mask = 1 - mask_tensor
    
    # Apply mask to compute weighted LPIPS
    # We focus on background regions by setting foreground regions to 0
    masked_tensor1 = tensor1 * bg_mask
    masked_tensor2 = tensor2 * bg_mask
    
    with torch.no_grad():
        distance = lpips_model(masked_tensor1, masked_tensor2)
    
    # Normalize by the background area
    bg_area = bg_mask.sum().item()
    if bg_area > 0:
        distance = distance * tensor1.numel() / bg_area
        
    return distance.item()

def compute_structure_dist(img1, img2):
    """
    Compute structural distance between two images using structural similarity index (SSIM)
    and gradient-based structural analysis
    
    Args:
        img1, img2: PIL Images to compare
    
    Returns:
        Dictionary containing various structural distance metrics
    """
    # Convert PIL images to numpy arrays
    img1_np = np.array(img1.convert('RGB'))
    img2_np = np.array(img2.convert('RGB'))
    
    # Convert to grayscale for structure analysis
    img1_gray = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)
    
    # Compute SSIM
    ssim_score, ssim_map = ssim(img1_gray, img2_gray, full=True)
    
    # Edge detection using Sobel filters
    img1_edges = cv2.Sobel(img1_gray, cv2.CV_64F, 1, 1, ksize=3)
    img2_edges = cv2.Sobel(img2_gray, cv2.CV_64F, 1, 1, ksize=3)
    
    # Normalize edge maps
    img1_edges = cv2.normalize(img1_edges, None, 0, 1, cv2.NORM_MINMAX)
    img2_edges = cv2.normalize(img2_edges, None, 0, 1, cv2.NORM_MINMAX)
    
    # Compute edge preservation ratio
    edge_diff = np.abs(img1_edges - img2_edges)
    edge_preservation = 1 - np.mean(edge_diff)
    
    # Compute gradient magnitude correlation
    gradient_correlation = np.corrcoef(img1_edges.flatten(), img2_edges.flatten())[0, 1]
    
    # Calculate overall structure distance (lower is better preservation)
    structure_distance = 1 - ((ssim_score + edge_preservation + max(0, gradient_correlation)) / 3)
    
    return {
        'ssim': ssim_score,
        'edge_preservation': edge_preservation,
        'gradient_correlation': gradient_correlation,
        'structure_distance': structure_distance
    }

def generate_foreground_mask(image, threshold=30):
    """
    Generate a simple foreground mask using edge detection and thresholding
    For more accurate results, you might want to use a segmentation model
    
    Args:
        image: PIL Image
        threshold: Threshold for edge detection
        
    Returns:
        Binary mask (numpy array) with 1 for foreground, 0 for background
    """
    # Convert to numpy array
    img_np = np.array(image.convert('RGB'))
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges
    edges = cv2.Canny(blurred, threshold, threshold * 3)
    
    # Dilate edges to create a foreground mask
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=3)
    
    # Fill holes in the mask
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    
    # Normalize to [0, 1]
    mask = mask / 255.0
    
    return mask

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_str', type=str, required=True)
    parser.add_argument('--random_seed', default=0)
    parser.add_argument('--task_name', type=str, default='cat2dog')
    parser.add_argument('--results_folder', type=str, default='output/test_cat')
    parser.add_argument('--num_deis_steps', type=int, default=50)
    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--xa_guidance', default=0.15, type=float)
    parser.add_argument('--negative_guidance_scale', default=5.0, type=float)
    parser.add_argument('--use_float_16', action='store_true')
    parser.add_argument('--compute_fg_mask', action='store_true', help='Generate foreground mask for BG-LPIPS')
    args = parser.parse_args()

    os.makedirs(args.results_folder, exist_ok=True)

    if args.use_float_16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # make the input noise map
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
    else:
        torch.manual_seed(args.random_seed)

    x = torch.randn((1,4,64,64), device=device)

    # 加载CLIP模型
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # 加载LPIPS模型
    lpips_model = lpips.LPIPS(net='alex').to(device)

    # Make the editing pipeline
    pipe = EditingPipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype).to(device)
    pipe.scheduler = DEISScheduler.from_config(pipe.scheduler.config)

    rec_pil, edit_pil = pipe(args.prompt_str, 
        num_inference_steps=args.num_deis_steps,
        x_in=x,
        edit_dir=construct_direction(args.task_name),
        guidance_amount=args.xa_guidance,
        guidance_scale=args.negative_guidance_scale,
        negative_prompt="" # use the empty string for the negative prompt
    )
    
    # 保存图像
    edit_path = os.path.join(args.results_folder, "edit.png")
    rec_path = os.path.join(args.results_folder, "reconstruction.png")
    edit_pil[0].save(edit_path)
    rec_pil[0].save(rec_path)
    
    # 计算CLIP scores
    rec_score = compute_clip_score(rec_pil[0], args.prompt_str, clip_model, clip_processor)
    edit_score = compute_clip_score(edit_pil[0], args.prompt_str, clip_model, clip_processor)
    
    # 计算BG-LPIPS
    if args.compute_fg_mask:
        # Generate foreground mask for more accurate BG-LPIPS
        fg_mask = generate_foreground_mask(rec_pil[0])
        # Save the mask for visualization
        mask_path = os.path.join(args.results_folder, "foreground_mask.png")
        Image.fromarray((fg_mask * 255).astype(np.uint8)).save(mask_path)
        bg_lpips_score = compute_bg_lpips(rec_pil[0], edit_pil[0], lpips_model, fg_mask)
    else:
        # Compute regular LPIPS without masking
        bg_lpips_score = compute_bg_lpips(rec_pil[0], edit_pil[0], lpips_model)
    
    # 计算Structure Distance
    structure_metrics = compute_structure_dist(rec_pil[0], edit_pil[0])
    
    # 打印评估指标
    print(f"\nEvaluation Metrics:")
    print(f"Reconstruction CLIP score: {rec_score:.4f}")
    print(f"Edit CLIP score: {edit_score:.4f}")
    print(f"BG-LPIPS score: {bg_lpips_score:.4f} (lower is better)")
    print(f"Structure Distance: {structure_metrics['structure_distance']:.4f} (lower is better)")
    print(f"SSIM: {structure_metrics['ssim']:.4f} (higher is better)")
    print(f"Edge Preservation: {structure_metrics['edge_preservation']:.4f} (higher is better)")
    print(f"Gradient Correlation: {structure_metrics['gradient_correlation']:.4f} (higher is better)")
    