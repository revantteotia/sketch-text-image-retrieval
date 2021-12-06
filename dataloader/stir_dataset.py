from PIL import Image
from omegaconf import OmegaConf
import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from utils.simple_tokenizer import SimpleTokenizer
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import numpy as np

from utils.util import load_from_yaml_file, read_json, load_config_file

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP mean, std
    ])

class STIRDataset(Dataset):

    def __init__(self, config, text_tokenizer, context_length=77, input_resolution=224, split="train"):
        super().__init__()

        self.config = config
        self.split = split
        self.images_directory_path = config.images_directory_path
        self.query_images_directory_path = config.query_images_directory_path

        if split == 'train':
            self.annotations = read_json(config.annotation_file_path_train)
        elif split == 'val':
            self.annotations = read_json(config.annotation_file_path_val)
        
        self.transform = _transform(input_resolution)
        self._tokenizer = text_tokenizer
        self.context_length = context_length

    def tokenize(self, text):
        sot_token = self._tokenizer.encoder["<|startoftext|>"]
        eot_token = self._tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + self._tokenizer.encode(text) + [eot_token]
        result = torch.zeros(self.context_length, dtype=torch.long)
        result[:len(tokens)] = torch.tensor(tokens)
        return result

    def get_all_texts(self):
        texts = [annotation["query_text"] for annotation in self.annotations]
        return texts

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # query_category = self.annotations[index]["query_category"]
        query_img_path = self.annotations[index]['query_sketch']
        query_text = self.annotations[index]["query_text"]
        target_img_id = self.annotations[index]["img_id"]

        query_img_path = os.path.join(self.query_images_directory_path, query_img_path)
        target_img_path = os.path.join(self.images_directory_path, str(target_img_id) + '.jpg')

        query_img = Image.open(query_img_path)
        target_img = Image.open(target_img_path)

        # cropping the target image
        x = self.annotations[index]['region']['x']
        y = self.annotations[index]['region']['y']
        h = self.annotations[index]['region']['h']
        w = self.annotations[index]['region']['w']
        region_area = (x, y, x+w, y+h) # left, top, right, bottom
        cropped_target_img = target_img.crop(region_area)

        # transforming text and images for model imput
        query_text_input = self.tokenize(query_text)
        query_img_input = self.transform(query_img)
        target_img_input = self.transform(cropped_target_img)

        return {
            'query_img_input' : query_img_input,
            'query_text_input' : query_text_input,
            'target_img_input' : target_img_input,
            'query_img_path' : query_img_path,
            'query_text' : query_text,
            'target_img_id' : target_img_id
        }

if __name__ == "__main__":
    # main functio to debug the dataset code only,

    DATA_CONFIG_PATH = '/home/trevant/DL4CV/dl4cv_project/configs/dataset_config.yaml'

    data_config = load_config_file(DATA_CONFIG_PATH)

    # TODO : 
    # 1. download CLIP source code from my repo and add it here (DONE)
    # 2. Somehow get clip trained weights for vision and text models separately and load them in the train.py

    # getting text tokenizer
    tokenizer = SimpleTokenizer()

    cqir_dataset = STIRDataset(data_config, tokenizer)

    # query_img_input, query_text_input, target_img_input = cqir_dataset[1]
    # print("query_img_input shape", query_img_input.shape)
    # print("query_text_input shape", query_text_input.shape)
    # print("target_img_input shape", target_img_input.shape)

