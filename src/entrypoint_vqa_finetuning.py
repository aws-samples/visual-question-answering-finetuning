import argparse
import logging
import os
import shutil
import sys

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
)

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.DEBUG)
LOGGER = logging.Logger("Finetune-VQA", level=logging.DEBUG)
HANDLER = logging.StreamHandler(sys.stdout)
HANDLER.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
LOGGER.addHandler(HANDLER)


class VQADataset(Dataset):
    def __init__(self, csv_file, root_dir, img_size=(1800, 2400)):
        self.attributes = pd.read_csv(csv_file)[["id", "Question", "Answer"]]
        self.root_dir = root_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.attributes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        item = {}
        item["id"] = str(self.attributes.iloc[idx, 0])
        item["question"] = (
            "Question: " + str(self.attributes.iloc[idx, 1]) + " Answer: "
        )
        item["answer"] = self.attributes.iloc[idx, 2]
        img_path = os.path.join(self.root_dir, (item["id"] + ".jpg"))

        item["image"] = Image.open(img_path)
        if item["image"].size != self.img_size:
            item["image"] = item["image"].resize(self.img_size)
        return (np.array(item["image"]), item["question"], item["answer"], item["id"])


def save_model(model, processor):
    LOGGER.info("Saving the model.")
    model.eval()
    model.save_pretrained(os.environ["SM_MODEL_DIR"])
    processor.save_pretrained(os.environ["SM_MODEL_DIR"])
    del model
    del processor
    # copy inference script and requirements.txt
    os.makedirs("/opt/ml/model/code", exist_ok=True)
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "inference.py"),
        "/opt/ml/model/code/inference.py",
    )
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "requirements.txt"),
        "/opt/ml/model/code/requirements.txt",
    )


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_name,
        device_map="auto",
        cache_dir="/tmp",
        load_in_8bit=True,
    )

    config = LoraConfig(
        r=8, # Lora attention dimension.
        lora_alpha=32, # the alpha parameter for Lora scaling.
        lora_dropout=0.05, # the dropout probability for Lora layers.
        bias="none", # the bias type for Lora.
        target_modules=["q", "v"],
    )

    model = get_peft_model(model, config)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    processor = AutoProcessor.from_pretrained(args.model_name)
    train_dataset = VQADataset(
        csv_file=f"{os.environ['SM_CHANNEL_INPUT_FILE']}/{args.file_name}",
        root_dir=os.environ["SM_CHANNEL_IMAGES"],
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    nan_flag = False
    for epoch in range(args.epochs):
        LOGGER.log(logging.DEBUG, ("=" * 30))
        LOGGER.log(logging.DEBUG, (f"Epoch: {epoch}"))
        epoch_train_loss = []
        for index, (img_tensor, str_question, str_answer, _) in enumerate(train_dataloader):
            if nan_flag:
                LOGGER.log(logging.DEBUG, (f"loss is nan after {epoch} epochs. breaking training loop"))
                break

            inputs = processor(
                images=img_tensor,
                text=str_question,
                text_target=str_answer,
                padding=True,
                return_tensors="pt",
            ).to(device)

            outputs = model(
                pixel_values=inputs["pixel_values"].to(device, dtype),
                input_ids=inputs["input_ids"],
                labels=inputs["labels"],
            )
            loss = outputs.loss

            if torch.isnan(loss).item():
                # https://github.com/huggingface/notebooks/issues/454
                LOGGER.log(logging.DEBUG, (f"loss is nan after {epoch} epochs"))
                nan_flag = True
                save_model(model, processor)
                break

            if index % 50 == 0:
                with torch.no_grad():
                    generated_output = model.generate(
                            pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"], max_length=20
                        )
                LOGGER.log(
                    logging.DEBUG,
                    (f"question: {processor.batch_decode(inputs['input_ids'], skip_special_tokens=True)[0]}"),
                )
                LOGGER.log(
                    logging.DEBUG,
                    (f"correct answer: {str_answer[0]}"),
                )
                LOGGER.log(
                    logging.DEBUG,
                    (f"prediction: {processor.batch_decode(generated_output, skip_special_tokens=True)[0]}"),
                )
                LOGGER.log(logging.DEBUG, ("=" * 30))

            epoch_train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        LOGGER.log(
            logging.DEBUG, (f"Epoch Loss: {np.asarray(epoch_train_loss).mean()}")
        )
    if not nan_flag:
        save_model(model, processor)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0025,
        help="learning rate (default: 0.0025)",
    )
    parser.add_argument(
        "--file-name",
        type=str,
        metavar="N",
        default="vqa_train.csv",
        help="filename of training dataset",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        metavar="N",
        default="Salesforce/blip2-flan-t5-xl",
        choices=[
            "Salesforce/blip2-flan-t5-xl",
            "Salesforce/blip2-flan-t5-xxl",
        ],
        help="from model id from Huggingface",
    )
    args = parser.parse_args()
    train(args)
