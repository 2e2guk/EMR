import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data._utils.collate import default_collate


class VisDialDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        self.text_processor = text_processor
        ann_path = ann_paths[0]

        with open(ann_path, 'r') as f:
            visdial_data = json.load(f)

        self.questions = visdial_data['data']['questions']
        self.dialogs_info = visdial_data['data']['dialogs']

        self.annotation = []
        for dialog in self.dialogs_info:
            for i, turn in enumerate(dialog['dialog']):
                self.annotation.append({
                    "image_id": dialog['image_id'],
                    "question_idx": turn['question'],
                    "answer_idx": turn['answer'],
                    "turn_id": i
                })

        self.embedding_root = vis_root

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_id = ann['image_id']
        turn_id = ann['turn_id']

        question_text = self.questions[ann['question_idx']]
        answer_text = question_text

        vision_embedding_path = os.path.join(self.embedding_root, f"{image_id}_vision.pt")
        text_embedding_path = os.path.join(self.embedding_root, f"{image_id}_q{turn_id}_text.pt")

        try:
            vision_embedding = torch.load(vision_embedding_path)
            text_embedding = torch.load(text_embedding_path)
        except FileNotFoundError:
            vision_embedding = torch.zeros(1024)
            text_embedding = torch.zeros(1024)

        return {
            "image_embedding": vision_embedding,
            "text_embedding": text_embedding,
            "turn_id": turn_id,
            "text_input": self.text_processor(question_text),
            "text_output": self.text_processor(answer_text),
        }

    def collater(self, samples):
        return default_collate(samples)