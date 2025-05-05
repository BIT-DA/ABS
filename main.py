import numpy as np
import torch
import yaml
import argparse
from helper import (
    accuracy,
    generate_weights,
    load_precomputed_features,
    set_seed
)
from clip import clip
from torchvision.transforms import v2 as T
from torchvision import datasets
from torch.nn import functional as F
from PIL import Image
import cv2
from transformers import AutoImageProcessor, ViTModel
from tqdm import tqdm
import random

def main(args):  
    device: str = "cuda"
    seed: int = args.seed
    num_workers: int = 8
    
    def custom_loader(path: str) -> torch.Tensor:
        img = datasets.folder.default_loader(path)
        W, H = img.size
        img_cv2 = np.array(img)
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
        image = img.resize((224, 224))
        augmented_imgs = [processor_dino(img, return_tensors="pt")["pixel_values"].squeeze(0)]
        augmented_imgs.extend(processor(img) for _ in range(1))
        attention_imgs_dino = processor_dino(img, return_tensors="pt")
        with torch.no_grad():
            image_attention_mh = model_dino(**attention_imgs_dino, output_attentions=True)
            image_attention_mh = image_attention_mh.attentions
        n_head = image_attention_mh[11].shape[1]
        attention_map = image_attention_mh[11][0, :, 0, 1:].reshape(n_head, -1).float()
        att_map = attention_map.mean(dim=0)

        top_k = args.top_k 
        att_map_flat = att_map.flatten() 
        topk_values, topk_indices = torch.topk(att_map_flat, top_k) 

        topk_probs = torch.softmax(topk_values / 0.03, dim=0) 

        num_samples = args.num_crops
        sampled_indices = torch.multinomial(topk_probs, num_samples, replacement=True) 
        sampled_patch_indices = topk_indices[sampled_indices] 
        crop_img_dino = []

        for sampled_index in sampled_patch_indices:
            i, j = sampled_index // crop_size, sampled_index % crop_size
            patch_x_min = int(j * patch_size * (W / (crop_size * patch_size))) 
            patch_y_min = int(i * patch_size * (H / (crop_size * patch_size))) 
            patch_x_max = min(patch_x_min + int(patch_size * (W / (crop_size * patch_size))), W)  
            patch_y_max = min(patch_y_min + int(patch_size * (H / (crop_size * patch_size))), H) 

            center_x = (patch_x_min + patch_x_max) // 2
            center_y = (patch_y_min + patch_y_max) // 2
            crop_width = random.randint(int(W * args.clip_crop_r1), int(W * args.clip_crop_r2))  
            crop_height = random.randint(int(H * args.clip_crop_r1), int(H * args.clip_crop_r2))  
            x_min = max(center_x - crop_width // 2, 0)
            y_min = max(center_y - crop_height // 2, 0)
            x_max = min(center_x + crop_width // 2, W)
            y_max = min(center_y + crop_height // 2, H)
            cropped_image = img.crop((x_min, y_min, x_max, y_max))
            crop_img_dino.append(cropped_image)
            augmented_imgs.extend(processor(cropped_image) for _ in range(1))

        return torch.stack(augmented_imgs)
    device = torch.device(device)
    print("Device:", device)
    print("num_workers:", num_workers)

    with open(file=f"cfgs/{args.dataset_name}.yaml") as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    set_seed(seed)

    model_size = hparams["model_size"]
    alpha = hparams["alpha"]
    n_samples = hparams["n_samples"]
    batch_size = hparams["batch_size"]
    data_path = hparams["data_path"]

    # load model
    print(f"Loading {model_size}")
    model, processor = clip.load(model_size, device=device)
    model.eval()
    model.requires_grad_(False)
    patch_size = 16
    crop_size = 14

    def load_ckpt(ckpt_id="facebook/dino-vitb16"):
        image_processor = AutoImageProcessor.from_pretrained(ckpt_id)
        model = ViTModel.from_pretrained(ckpt_id).eval()
        return model, image_processor

    ckpt_id = "facebook/dino-vitb16"

    model_dino, processor_dino = load_ckpt(ckpt_id)
    model_dino.eval()
    model_dino.requires_grad_(False)


    precomputed_features,target,image_features,= load_precomputed_features(model,args,processor,dataset_name=args.dataset_name,model_size=model_size,alpha=alpha,n_samples=n_samples,batch_size=batch_size,num_workers=num_workers,data_path=data_path,custom_loader=custom_loader,device=device,layer1=args.layer1,layer2=args.layer2)

    max_size = precomputed_features.size(1)
    image_features = image_features.to(device)

    results = {}
    with torch.no_grad():
        methods = hparams["methods"]
        for method in methods:
            method = list(method.values())[0]
            method_name = method["name"]
            method_enabled = method["enabled"]

            text_scale = (
                torch.exp(torch.tensor(method["text_scale"])).to(device)
                if "text_scale" in method
                else None
            )
            image_scale = (
                torch.exp(torch.tensor(method["image_scale"])).to(device)
                if "image_scale" in method
                else None
            )

            if method_enabled:
                zeroshot_weights = generate_weights(
                    method_name,
                    model=model,
                    dataset_name=args.dataset_name,
                    tt_scale=text_scale,
                    device=device,
                )
            
                zeroshot_weights = zeroshot_weights.to(image_features.dtype)
            else:
                continue

            
            if method_name != "ours":
                logits = image_features.squeeze(1) @ zeroshot_weights
                baseline_acc = accuracy(
                    logits, target, image_features.size(0), args.dataset_name
                )
                print(f"{method_name}: {baseline_acc:.2f}\n")
                results[method_name] = round(baseline_acc, 2)

            if method_name == "ours":
                acc_list = []
                patch_num = hparams["patch_n"]
                zeroshot_weights = zeroshot_weights.permute(1, 0, 2) 
                print(f"n_run: {hparams['n_run']}")
                for i in range(hparams["n_run"]):
                    random_indices = torch.randint(0, max_size, (patch_num,))
                    sampled_features = precomputed_features
                    patch_embeds = sampled_features[:, :, :-1] 
                    patch_weights = sampled_features[:, :, -1]
                    del sampled_features
                    logits_sum = []
                    logits_total = []
                    batch_size = 100
                    total_size, crop_num, embed_dim = patch_embeds.shape
                    num_classes, num_descriptions, embed_dim = zeroshot_weights.shape
                    num_batches = (total_size + batch_size - 1) // batch_size  
                    logits_total = []
                    for batch_idx in tqdm(range(num_batches)):
                        start_idx = batch_idx * batch_size
                        end_idx = min((batch_idx + 1) * batch_size, total_size)
                        patch_weights_batch = patch_weights[start_idx:end_idx]
                        patch_embeds_batch = patch_embeds[start_idx:end_idx]  
                        patch_embeds_flat = patch_embeds_batch.reshape(-1, embed_dim) 
                        zeroshot_weights_flat = zeroshot_weights.reshape(-1, embed_dim) 
                        similarity_matrix_flat = torch.matmul(patch_embeds_flat, zeroshot_weights_flat.t()) 
                        similarity_matrix_flat = similarity_matrix_flat / 0.03
                        similarity_matrix = similarity_matrix_flat.reshape(end_idx - start_idx, crop_num, num_classes, num_descriptions)
                        similarity_matrix = similarity_matrix.view(end_idx - start_idx, crop_num, -1)
                        log_softmax_matrix = similarity_matrix.log_softmax(dim=-1)  
                        similarity_matrix_soft = log_softmax_matrix.exp()
                        weighted_similarity_matrix = similarity_matrix_soft * similarity_matrix  
                        weighted_similarity_matrix = weighted_similarity_matrix.reshape(end_idx - start_idx, crop_num, num_classes, num_descriptions)
                        logits_batch_crop_class = weighted_similarity_matrix.sum(dim=-1)  
                        w_i = (patch_weights_batch * image_scale).softmax(-1).unsqueeze(-1) 
                        logits_batch = (logits_batch_crop_class).sum(dim=1) 
                        logits_total.append(logits_batch)
        
                    logits = torch.cat(logits_total, dim=0)  
                    acc_list.append(
                        accuracy(logits, target, patch_embeds.size(0), args.dataset_name)
                    )

                mean = np.mean(acc_list)
                std = np.std(acc_list)
                with open('results.txt', 'a') as f:
                    f.write(f"{method_name} {args.dataset_name}: {mean:.2f}+-{std:.2f}\n  ")
                    for acc in acc_list:
                        f.write(f"{acc}\n")
                    f.write('----------------\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script using argparse")
    parser.add_argument('--dataset_name', type=str, default='oxford_pet', help='Name of datasets')
    parser.add_argument('--patch_size', type=int, default=14, help='Size of layer1')
    parser.add_argument('--num_crops', type=int, default=50, help='Number of crops')
    parser.add_argument('--top_k', type=int, default=20, help='Topk values')
    parser.add_argument('--layer1', type=int, default=11, help='Size of layer1')
    parser.add_argument('--layer2', type=int, default=11, help='Size of layer2')
    parser.add_argument('--clip_crop_r1', type=float, default=0.6, help='Clip crop ratio 1')
    parser.add_argument('--clip_crop_r2', type=float, default=0.9, help='Clip crop ratio 2')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    args = parser.parse_args()
    main(args)
