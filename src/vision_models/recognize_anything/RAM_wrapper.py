import json
from typing import List

import cv2
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from ram import get_transform
from ram import inference_ram_openset as inference
from ram.models import ram_plus
from ram.utils import build_openset_llm_label_embedding
from torch import nn

from vision_models.utils import get_checkpoint_path, get_checkpoints_names

IMAGE_SIZE = 384


class RAM_wrapper:
    def __init__(self, objects_descriptions_file_path: str, checkpoint_name: str = "ram_plus_swin_large_14m") -> None:
        valid_names = get_checkpoints_names()
        assert checkpoint_name in valid_names
        self.checkpoint_path = get_checkpoint_path(checkpoint_name)

        # Building embeddings
        self.openset_label_embedding, self.openset_categories = build_openset_llm_label_embedding(
            json.load(open(objects_descriptions_file_path, "rb"))
        )
        self.device = torch.device("cuda")  # if torch.cuda.is_available() else "cpu")
        self.transform = get_transform(image_size=IMAGE_SIZE)
        self.model = ram_plus(pretrained=self.checkpoint_path, image_size=IMAGE_SIZE, vit="swin_l")

        self.model.tag_list = np.array(self.openset_categories)
        self.model.label_embed = nn.Parameter(self.openset_label_embedding.float())
        self.model.num_class = len(self.openset_categories)

        self.model.class_threshold = torch.ones(self.model.num_class) * 0.5
        self.model.eval()
        self.model = self.model.to(self.device)

    def infer(self, im: npt.NDArray[np.uint8]) -> List[str]:
        im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
        im = Image.fromarray(im)
        im = self.transform(im).unsqueeze(0).to(self.device)
        res: str = inference(im, self.model)

        labels: List[str] = res.split("|")
        for i in range(len(labels)):
            labels[i] = labels[i].strip()

        return labels
