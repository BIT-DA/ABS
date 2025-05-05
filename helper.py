import os
import random
import numpy as np
import torch
import json
import pickle
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from clip import clip
from torchvision.datasets import ImageNet, ImageFolder, Places365
from clip import clip
from my_datasets import *
from utils import (
    openai_imagenet_classes,
    imagenet_classes,
    imagenet_a_lt,
    imagenet_r_lt,
)


def load_json(filename):
    if not filename.endswith(".json"):
        filename += ".json"
    with open(filename, "r") as fp:
        return json.load(fp)


def set_seed(seed):
    print(f"Setting seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(data_path, dataset_name, custom_loader):
    data_path = data_path

    if dataset_name == MyDataset.ImageNet:
        dataset = ImageNet(
            data_path,
            split="val",
            transform=None,
            loader=custom_loader,
        )

    elif dataset_name == MyDataset.ImageNetV2:
        dataset = ImageNetV2Dataset(
            location=data_path,
            transform=None,
            loader=custom_loader,
        )

    elif dataset_name == MyDataset.ImageNetR:
        dataset = ImageFolder(
            root=data_path,
            transform=None,
            loader=custom_loader,
        )

    elif dataset_name == MyDataset.ImageNetS:
        dataset = ImageFolder(
            root=data_path,
            transform=None,
            loader=custom_loader,
        )

    elif dataset_name == MyDataset.ImageNetA:
        dataset = ImageFolder(
            root=data_path,
            transform=None,
            loader=custom_loader,
        )

    elif dataset_name == MyDataset.CUB:
        dataset = CUBDataset(
            data_path,
            train=False,
            transform=None,
            loader=custom_loader,
        )

    elif dataset_name == MyDataset.Food101:
        dataset = Food101(
            data_path,
            transform=None,
            loader=custom_loader,
            split="test",
            download=False,
        )

    elif dataset_name == MyDataset.OxfordIIITPet:
        dataset = OxfordIIITPet(
            data_path,
            transform=None,
            split="test",
            loader=custom_loader,
        )

    elif dataset_name == MyDataset.Place365:
        dataset = Places365(
            data_path,
            transform=None,
            loader=custom_loader,
            download=False,
            split="val",
            small=False,
        )

    elif dataset_name == MyDataset.DTD:
        dataset = DTD(
            data_path,
            loader=custom_loader,
            split="test",
            download=False,
        )

    return dataset


def wordify(string):
    word = string.replace("_", " ")
    return word


def load_classes(dataset_name):
    with open(
        f"features/{dataset_name}/{dataset_name}.json",
        "r",
    ) as f:
        classes = json.load(f)

    wordify_classes = []
    for c in classes:
        wordify_classes.append(wordify(c))

    return wordify_classes
def generate_weights(
    method,
    model,
    dataset_name,
    tt_scale=None,
    device=None,
):
    templates = None
    make_sentence = False
    is_template = True

    if dataset_name.startswith(MyDataset.ImageNet):
        classes = (
            openai_imagenet_classes
            if method in ["clip-d", "waffle"]
            else imagenet_classes
        )
    else:
        classes = load_classes(dataset_name)

    print(f"Creating {method} text embeddings...")

    if method != "clip":
        if method == "ours":
            load_file = "cupl"
        elif method == "cupl":
            load_file = "cupl"
        elif method == "waffle":
            load_file = "clip-d"
        else:
            load_file = method

        with open(f"prompts/{dataset_name}/{load_file}.json") as f:
            templates = json.load(f)

        if method in ["waffle", "clip-d", "cupl", "ours"]:
            is_template = False

        if method == "clip-d":
            make_sentence = True

        if method == "waffle":
            templates = construct_random(templates)

    zeroshot_weights = zeroshot_classifier(
        model,
        classes,
        templates,
        is_template,
        make_sentence,
        tt_scale,
        device,
    )

    return zeroshot_weights

from transformers import AutoImageProcessor, ViTModel

def load_ckpt(ckpt_id="facebook/dino-vits16"):
    """Loads a pretrained model along with its processor class."""
    image_processor = AutoImageProcessor.from_pretrained(ckpt_id)
    model = ViTModel.from_pretrained(ckpt_id).eval()
    return model, image_processor


def load_precomputed_features(
    model,
    args,
    processor,
    dataset_name: str,
    model_size: str,
    alpha: float,
    n_samples: int,
    batch_size: int,
    num_workers: int,
    data_path: str,
    custom_loader: callable,
    device: torch.device,
    layer1: int,
    layer2: int,
):
    save_file = (dataset_name + "-" + model_size).replace("/", "-")
    save_root = f"features/{dataset_name}"

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    filename = os.path.join(save_root, f"{save_file}-crop_fea-1layer-{args.num_crops}-{args.top_k}-with-dino2.pkl")

    if os.path.exists(filename):
        print(f"Loading {filename}...")
        load_res = pickle.load(open(filename, "rb"))
    else:
        print(f"File {filename} not found, precomputing features...")
        dataset = load_dataset(
            data_path=data_path,
            dataset_name=dataset_name,
            custom_loader=custom_loader,
        )

        dataloader = DataLoader(
            dataset,
            batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
        )

     
        precomputed_features = []
        image_features_tensor = []
        target = []
        patch_size = 32
        num_crop = args.num_crops
        ckpt_id = "facebook/dino-vitb16"
        model_dino, processor_dino = load_ckpt(ckpt_id)
        model_dino.eval()
        model_dino.requires_grad_(False)
        model_dino.to(device)

        with torch.no_grad():
            for batch in tqdm(dataloader):
                images, labels = [p.to(device) for p in batch]
                b, ns = images.shape[:2]
                images_dino = images[:,0,:,:,:]
                images_crop = images[:,2:,:,:,:]
                images = images[:,1,:,:,:]
                images_crop = images_crop.flatten(0, 1)
                image_attention_mh = model_dino(pixel_values=images_dino, output_attentions=True)
                image_attention_mh = image_attention_mh.attentions
                n_head = image_attention_mh[layer1].shape[1]
                attention_map = image_attention_mh[layer1][:, :, 0, 1:].reshape(b, n_head, -1).float()
                image_features = model.encode_image_attention(images, args,target_layer=[layer2], attn_weights=attention_map)
                image_features = F.normalize(image_features)
                image_features = image_features.view(b, num_crop+1, -1) 
                image_features_crop = model.encode_image(images_crop)
                image_features_crop = F.normalize(image_features_crop)
                image_features_crop = image_features_crop.view(b, num_crop, -1)
                patch_features = torch.cat((image_features[:, 1:], image_features_crop), dim=1)
                image_features = image_features[:, :1]
                weight_image = (image_features * patch_features).sum(
                    dim=-1, keepdim=True
                )
                patch_with_weights = torch.cat([patch_features, weight_image], -1)
                precomputed_features.append(patch_with_weights)
                target.append(labels)
                image_features_tensor.append(image_features.squeeze(1))

        load_res = {
            "patches": torch.cat(precomputed_features, dim=0),
            "images": torch.cat(image_features_tensor, dim=0),
            "labels": torch.cat(target, dim=0),
        }

        os.makedirs(save_root, exist_ok=True)
        pickle.dump(load_res, open(filename, "wb"))

    precomputed_features = load_res["patches"].to(device)
    target = load_res["labels"].to(device)
    image_features_tensor = load_res["images"].to(device)

    return precomputed_features, target, image_features_tensor


def make_descriptor_sentence(descriptor):
    if descriptor.startswith("a") or descriptor.startswith("an"):
        return f"which is {descriptor}"
    elif (
        descriptor.startswith("has")
        or descriptor.startswith("often")
        or descriptor.startswith("typically")
        or descriptor.startswith("may")
        or descriptor.startswith("can")
    ):
        return f"which {descriptor}"
    elif descriptor.startswith("used"):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"


def zeroshot_classifier(
    model,
    textnames,
    templates=None,
    is_template=True,
    make_sentence=False,
    tt_scale=None,
    device=None,
):
    with torch.no_grad():
        zeroshot_weights = []
        for i in tqdm(range(len(textnames))):
            if not is_template:
                texts = []
                for t in templates[textnames[i]]:
                    if make_sentence:
                        desc_sen = make_descriptor_sentence(t)
                        texts.append(f"{textnames[i]}, {desc_sen}")
                    else:
                        texts.append(t)
            elif templates:
                texts = [template.format(textnames[i]) for template in templates]
            else:
                texts = [f"a photo of a {textnames[i]}."]

            if i == 0:
                print(texts)

            if tt_scale is not None:
                label = f"a photo of a {textnames[i]}."
                label_tokens = clip.tokenize(label, truncate=True).to(device)
                label_embeddings = model.encode_text(label_tokens)
                label_embeddings /= label_embeddings.norm(dim=-1, keepdim=True)
                texts = texts[:50]

            texts_tensor = clip.tokenize(texts, truncate=True).to(device)
            class_embeddings = model.encode_text(texts_tensor)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

            if tt_scale is not None:
                weight = class_embeddings @ label_embeddings.T
                weight = (weight * tt_scale).softmax(dim=0)
                class_embedding = class_embeddings
            else:
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def construct_random(gpt3_prompts):
    """
    reference: https://github.com/ExplainableML/WaffleCLIP.git
    """
    key_list = list(gpt3_prompts.keys())
    descr_list = [list(values) for values in gpt3_prompts.values()]
    descr_list = np.array([x for y in descr_list for x in y])
    structured_descriptor_builder = (
        lambda item, cls: f"A photo of a {wordify(cls)}, {make_descriptor_sentence(item)}."
    )
    word_list = pickle.load(open("features/word_list.pkl", "rb"))
    avg_num_words = int(
        np.max(
            [
                np.round(np.mean([len(wordify(x).split(" ")) for x in key_list])),
                1,
            ]
        )
    )
    avg_word_length = int(
        np.round(
            np.mean(
                [np.mean([len(y) for y in wordify(x).split(" ")]) for x in key_list]
            )
        )
    )
    word_list = [x[:avg_word_length] for x in word_list]
    character_list = [x.split(" ") for x in descr_list]
    character_list = [
        x.replace(",", "").replace(".", "")
        for x in np.unique([x for y in character_list for x in y])
    ]
    character_list = np.unique(list("".join(character_list)))
    num_spaces = (
        int(np.round(np.mean([np.sum(np.array(list(x)) == " ") for x in key_list]))) + 1
    )
    num_chars = int(
        np.ceil(np.mean([np.max([len(y) for y in x.split(" ")]) for x in key_list]))
    )

    num_chars += num_spaces - num_chars % num_spaces
    sample_key = ""
    for s in range(num_spaces):
        for _ in range(num_chars // num_spaces):
            sample_key += "a"
        if s < num_spaces - 1:
            sample_key += " "

    gpt3_prompts = {key: [] for key in gpt3_prompts.keys()}

    for key in key_list:
        for _ in range(15):
            base_word = ""
            for a in range(avg_num_words):
                base_word += np.random.choice(word_list, 1, replace=False)[0]
                if a < avg_num_words - 1:
                    base_word += " "
            gpt3_prompts[key].append(structured_descriptor_builder(base_word, key))
            noise_word = ""
            use_key = sample_key if len(key) >= len(sample_key) else key
            for c in sample_key:
                if c != " ":
                    noise_word += np.random.choice(character_list, 1, replace=False)[0]
                else:
                    noise_word += ", "
            gpt3_prompts[key].append(structured_descriptor_builder(noise_word, key))

    match_key = np.random.choice(key_list)
    gpt3_prompts = {key: gpt3_prompts[match_key] for key in key_list}
    for key in gpt3_prompts:
        gpt3_prompts[key] = [
            x.replace(wordify(match_key), wordify(key)) for x in gpt3_prompts[key]
        ]

    return gpt3_prompts


def accuracy(output, target, n, dataset_name):
    if dataset_name.startswith(MyDataset.ImageNetA):
        _, pred = output[:, imagenet_a_lt].max(1)
    elif dataset_name.startswith(MyDataset.ImageNetR):
        _, pred = output[:, imagenet_r_lt].max(1)
    else:
        _, pred = output.max(1)
    correct = pred.eq(target)
    return float(correct.float().sum().cpu().numpy()) / n * 100
