import os

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import torchmetrics
import datasets
import transformers


class CrossEncoderModel(pl.LightningModule):
    def __init__(self, encoder: str, num_classes, num_warmup_steps, lr):
        super().__init__()
        self.save_hyperparameters()
        self.num_warmup_steps = num_warmup_steps
        self.lr = lr
        self.encoder = encoder
        embd_size = self.encoder.config.hidden_size
        self.classificator = torch.nn.Linear(embd_size, num_classes)

        self.loss = torch.nn.CrossEntropyLoss()
        metrics = torchmetrics.MetricCollection(
            {
                "acc": torchmetrics.Accuracy(task="multiclass", num_classes=2),
                "f1": torchmetrics.F1Score(task="multiclass", num_classes=2),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, input_ids, token_type_ids, attention_mask):
        model_output = self.encoder(input_ids, token_type_ids, attention_mask)[0]
        sentence_embd = self.mean_pooling(model_output, attention_mask)
        preds = self.classificator(sentence_embd)
        return preds

    def training_step(self, batch: dict, batch_idx):
        preds = self(**batch["pair"])
        loss = self.loss(preds, batch["label"])
        self.train_metrics(preds, batch["label"])

        self.log("train_loss", loss.item(), sync_dist=True, on_step=True, on_epoch=True)
        self.log_dict(self.train_metrics, sync_dist=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: dict, batch_idx):
        preds = self(**batch["pair"])
        loss = self.loss(preds, batch["label"])
        metrics = self.val_metrics(preds, batch["label"])

        self.log("val_loss", loss.item(), sync_dist=True, on_step=False, on_epoch=True)
        self.log_dict(metrics, sync_dist=True, on_step=False, on_epoch=True)

    def test_step(self, batch: dict, batch_idx):
        preds = self(**batch["pair"])
        metrics = self.test_metrics(preds, batch["label"])

        self.log_dict(metrics, sync_dist=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [
            {"scheduler": scheduler, "name": "cosine_scheduler", "interval": "step"}
        ]

    @staticmethod
    def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor):
        token_embeddings = model_output
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.shape).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
