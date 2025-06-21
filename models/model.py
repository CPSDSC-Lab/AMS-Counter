import argparse
import numpy as np
import os
import random
import math
from PIL import Image
import torch
import torch.nn.functional as F
from typing import List, Dict, Any

import util.misc as misc
from models.contrastive_loss import ContrastiveLoss
from models import clip_count
from pytorch_lightning import LightningModule
import einops
import cv2
from util.constant import SCALE_FACTOR


class Model(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # 如果args是dict，则转换为Namespace
        if self.args is not None and type(self.args) is dict:
            self.args = argparse.Namespace(**self.args)

        self.save_hyperparameters(args)

        self.model = clip_count.CLIPCount(
            fim_depth=self.args.decoder_depth,
            fim_num_heads=self.args.decoder_head,
            use_coop=self.args.use_coop,
            use_vpt=self.args.use_vpt,
            coop_width=self.args.coop_width,
            vpt_width=self.args.vpt_width,
            vpt_depth=self.args.vpt_depth,
            backbone=self.args.backbone,
            use_fim=self.args.use_fim,
            use_mixed_fim=self.args.use_mixed_fim,
            unfreeze_vit=self.args.unfreeze_vit,
        )
        self.loss = F.mse_loss
        self.contrastive_loss = ContrastiveLoss(self.model.text_encoder, 0.07, self.args.noise_text_ratio,
                                                self.args.normalize_contrast)
        self.neg_prompt_embed = None

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def training_step(self, batch, batch_idx):

        samples, gt_density, prompt_gt = batch

        output, extra_out = self.model(samples, prompt_gt, return_extra=True, coop_require_grad=True)

        if not self.args.use_contrast:
            prompt_gt = [f"a photo of {p}" for p in prompt_gt]

        # Compute loss function
        mask = np.random.binomial(n=1, p=0.8, size=[384, 384])
        masks = np.tile(mask, (output.shape[0], 1))
        masks = masks.reshape(output.shape[0], 384, 384)
        masks = torch.from_numpy(masks).to(self.device)

        loss = self.loss(output, gt_density)
        loss = (loss * masks / (384 * 384)).sum() / output.shape[0]

        sim_map = extra_out['sim_map']
        if self.args.use_contrast and self.current_epoch <= self.args.contrast_pre_epoch:
            text_embedding = extra_out['text_embedding']  # [B,1, 512]
            if self.args.contrast_pos == "pre":
                patch_embedding = extra_out['patch_embedding']  # [B, 196, 512]
            elif self.args.contrast_pos == "post":
                patch_embedding = extra_out['pixel_text_matching_map']  # [B, 512, 32, 32]
            img_embedding = extra_out['x_cls']  # [B, 1, 768]
            contrast_loss = self.contrastive_loss(patch_embedding, img_embedding, text_embedding, self.neg_prompt_embed,
                                                  gt_density.detach().clone(), prompt_gt, sim_map)
            loss = self.args.w_contrast * contrast_loss
            self.log('train_loss_contrast', contrast_loss)

        self.log('train_loss', loss)

        # Update information of MAE and RMSE
        batch_mae = 0
        batch_rmse = 0
        gt_sum = 0

        for i in range(output.shape[0]):
            pred_cnt = torch.sum(output[i] / SCALE_FACTOR).item()
            gt_cnt = torch.sum(gt_density[i] / SCALE_FACTOR).item()
            cnt_err = abs(pred_cnt - gt_cnt)
            gt_sum += gt_cnt
            batch_mae += cnt_err
            batch_rmse += cnt_err ** 2
        batch_mae /= output.shape[0]
        batch_rmse /= output.shape[0]
        batch_rmse = math.sqrt(batch_rmse)
        # loss = loss / gt_sum
        self.log('train_mae', batch_mae)
        self.log('train_rmse', batch_rmse)

        return loss

    def on_train_epoch_end(self):
        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        samples, gt_density, prompt = batch
        if not self.args.use_contrast:
            prompt = [f"a photo of {p}" for p in prompt]

        output, extra_out = self.model(samples, prompt, return_extra=True)

        # Update information of MAE and RMSE
        batch_mae = []
        batch_rmse = []
        pred_cnts = []
        gt_cnts = []
        for i in range(output.shape[0]):
            pred_cnt = torch.sum(output[i] / SCALE_FACTOR).item()  # SCALE_FACTOR is the scaling factor as CounTR uses
            gt_cnt = torch.sum(gt_density[i] / SCALE_FACTOR).item()
            cnt_err = abs(pred_cnt - gt_cnt)
            batch_mae.append(cnt_err)
            batch_rmse.append(cnt_err ** 2)
            pred_cnts.append(pred_cnt)
            gt_cnts.append(gt_cnt)

        # log the image
        img_log = samples[0].detach().cpu().numpy()
        pred_density = output[0].detach().cpu().numpy()
        pred_log_rgb = cv2.applyColorMap(np.uint8(255 * pred_density), cv2.COLORMAP_JET)
        pred_log_rgb = np.transpose(pred_log_rgb, (2, 0, 1))
        gt_density_log = gt_density[0].detach().cpu().numpy()
        gt_log_rgb = cv2.applyColorMap(np.uint8(255 * gt_density_log), cv2.COLORMAP_JET)
        gt_log_rgb = np.transpose(gt_log_rgb, (2, 0, 1))

        pred_density = einops.repeat(pred_density, 'h w -> c h w', c=3)
        pred_density = pred_density / pred_density.max()  # normalize
        heatmap_pred = 0.33 * img_log + 0.67 * pred_density
        gt_density_log = einops.repeat(gt_density_log, 'h w -> c h w', c=3)
        heatmap_gt = 0.33 * img_log + 0.67 * gt_density_log

        sim_map = extra_out['sim_map_16x'][0].detach().cpu().numpy()
        sim_map = (sim_map - np.min(sim_map)) / (np.max(sim_map) - np.min(sim_map))
        sim_map_rgb = cv2.applyColorMap(np.uint8(255 * sim_map[0, :, :]), cv2.COLORMAP_JET)
        sim_map_rgb = np.transpose(sim_map_rgb, (2, 0, 1))

        self.validation_step_outputs.append(
            {"mae": batch_mae, "rmse": batch_rmse, "img": img_log, "pred": pred_log_rgb, "gt": gt_log_rgb,
             "heatmap_pred": heatmap_pred, "heatmap_gt": heatmap_gt, "prompt": prompt[0], "pred_cnts": pred_cnts,
             "gt_cnts": gt_cnts, 'sim_map': sim_map_rgb})

        return {"mae": batch_mae, "rmse": batch_rmse, "img": img_log, "pred": pred_log_rgb, "gt": gt_log_rgb,
                "heatmap_pred": heatmap_pred, "heatmap_gt": heatmap_gt, "prompt": prompt[0], "pred_cnts": pred_cnts,
                "gt_cnts": gt_cnts}

    def on_validation_epoch_end(self):
        all_mae = []
        all_rmse = []

        outputs = self.validation_step_outputs

        for output in outputs:
            all_mae += output["mae"]
            all_rmse += output["rmse"]
        val_mae = np.mean(all_mae)
        val_rmse = np.sqrt(np.mean(all_rmse))
        self.log('val_mae', val_mae)
        self.log('val_rmse', val_rmse)

        # log the image
        idx = random.randint(0, len(outputs) - 1)
        img = outputs[idx]["img"]
        pred = outputs[idx]["pred"]
        gt = outputs[idx]["gt"]
        heatmap_pred = outputs[idx]["heatmap_pred"]
        heatmap_gt = outputs[idx]["heatmap_gt"]
        prompt = outputs[idx]["prompt"]
        pred_cnts = outputs[idx]["pred_cnts"]
        gt_cnts = outputs[idx]["gt_cnts"]
        pred_gt = "pred: {:.2f} gt: {:.2f}".format(pred_cnts[0], gt_cnts[0])
        sim_map = outputs[idx]["sim_map"]

        # self.logger.experiment.add_image("sim_map", sim_map, self.current_epoch)
        # self.logger.experiment.add_image("val_img", img, self.current_epoch)
        # self.logger.experiment.add_image("density_pred", pred, self.current_epoch)
        # self.logger.experiment.add_image("density_gt", gt, self.current_epoch)
        # self.logger.experiment.add_image("overlay_pred", heatmap_pred, self.current_epoch)
        # self.logger.experiment.add_image("overlay_gt", heatmap_gt, self.current_epoch)
        # self.logger.experiment.add_text("prompt", prompt, self.current_epoch)
        # self.logger.experiment.add_text("count", pred_gt, self.current_epoch)

        self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        if self.args.dataset_type == 'FSC' or self.args.dataset_type == "COCO":
            image, gt_density, prompt = batch
        elif self.args.dataset_type == "CARPK":
            image, gt_cnt = batch
            gt_cnt = gt_cnt.item()
            prompt = ["car" for _ in range(image.shape[0])]
            gt_density = torch.zeros(image.shape[0], image.shape[2], image.shape[3])
        elif self.args.dataset_type == "ShanghaiTech":
            image, gt_cnt = batch
            gt_cnt = gt_cnt.item()
            prompt = ["people" for _ in range(image.shape[0])]
            gt_density = torch.zeros(image.shape[0], image.shape[2], image.shape[3])

        assert image.shape[0] == 1, "only support inference one image at a time"
        raw_h, raw_w = image.shape[2:]

        patches, _ = misc.sliding_window(image, stride=128)
        # covert to batch
        patches = torch.from_numpy(patches).float().to(self.device)
        prompt = np.repeat(prompt, patches.shape[0], axis=0)
        output, _ = self.model(patches, prompt)
        output.unsqueeze_(1)
        output = misc.window_composite(output, stride=128)
        output = output.squeeze(1)
        # crop to original width
        output = output[:, :, :raw_w]

        # Update information of MAE and RMSE
        batch_mae = []
        batch_rmse = []
        pred_cnts = []
        gt_cnts = []

        pred_cnt = torch.sum(output[0] / SCALE_FACTOR).item()
        if self.args.dataset_type == "FSC" or self.args.dataset_type == "COCO":
            gt_cnt = torch.sum(gt_density[0] / SCALE_FACTOR).item()
        cnt_err = abs(pred_cnt - gt_cnt)
        batch_mae.append(cnt_err)
        batch_rmse.append(cnt_err ** 2)
        pred_cnts.append(pred_cnt)
        gt_cnts.append(gt_cnt)

        # log the image
        img_log = image[0].detach().cpu().numpy()
        pred_density = output[0].detach().cpu().numpy()
        pred_log_rgb = cv2.applyColorMap(np.uint8(255 * pred_density), cv2.COLORMAP_JET)
        pred_log_rgb = np.transpose(pred_log_rgb, (2, 0, 1))
        gt_density_log = gt_density[0].detach().cpu().numpy()
        gt_log_rgb = cv2.applyColorMap(np.uint8(255 * gt_density_log), cv2.COLORMAP_JET)
        gt_log_rgb = np.transpose(gt_log_rgb, (2, 0, 1))

        pred_density = einops.repeat(pred_density, 'h w -> c h w', c=3)
        pred_density = pred_density / pred_density.max()  # normalize
        heatmap_pred = img_log
        heatmap_pred = 0.33 * img_log + 0.67 * pred_density
        gt_density_log = einops.repeat(gt_density_log, 'h w -> c h w', c=3)
        heatmap_gt = img_log

        # log qualitative results
        if self.args.log_test_img:
            if cnt_err < 5:
                # log density
                log_dir = "out/good_density/"
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                name = "good_{}_{:.2f}_gt_{:.2f}.jpg".format(prompt[0], pred_cnt, gt_cnt)
                pred_density_write = 1. - pred_density[0]
                pred_density_write = cv2.applyColorMap(np.uint8(255 * pred_density_write), cv2.COLORMAP_JET)
                img = Image.fromarray(np.uint8(pred_density_write))
                img.save(log_dir + name)

                log_dir = "out/good_pred/"
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                # log overlay
                name = "good_{}_{:.2f}_gt_{:.2f}.jpg".format(prompt[0], pred_cnt, gt_cnt)
                pred_density_write = pred_density_write / 255.
                img_write = 0.33 * np.transpose(img_log, (1, 2, 0)) + 0.67 * pred_density_write
                img = Image.fromarray(np.uint8(255 * img_write))
                img.save(log_dir + name)

            if cnt_err > 100:
                # save image, overlaied
                # log density
                name = "good_{}_{:.2f}_gt_{:.2f}.jpg".format(prompt[0], pred_cnt, gt_cnt)
                pred_density_write = 1. - pred_density[0]
                pred_density_write = cv2.applyColorMap(np.uint8(255 * pred_density_write), cv2.COLORMAP_JET)

                log_dir = "debug/bad_pred/"
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                name = "bad_{}_{:.2f}_gt_{:.2f}.jpg".format(prompt[0], pred_cnt, gt_cnt)
                pred_density_write = pred_density_write / 255.
                img_write = 0.33 * np.transpose(img_log, (1, 2, 0)) + 0.67 * pred_density_write
                img = Image.fromarray(np.uint8(255 * img_write))
                img.save(log_dir + name)

        self.test_step_outputs.append(
            {"mae": batch_mae, "rmse": batch_rmse, "img": img_log, "pred": pred_log_rgb, "gt": gt_log_rgb,
             "heatmap_pred": heatmap_pred, "heatmap_gt": heatmap_gt, "prompt": prompt[0], "pred_cnts": pred_cnts,
             "gt_cnts": gt_cnts})

        return {"mae": batch_mae, "rmse": batch_rmse, "img": img_log, "pred": pred_log_rgb, "gt": gt_log_rgb,
                "heatmap_pred": heatmap_pred, "heatmap_gt": heatmap_gt, "prompt": prompt[0], "pred_cnts": pred_cnts,
                "gt_cnts": gt_cnts}

    def on_test_epoch_end(self):
        all_mae = []
        all_rmse = []

        outputs = self.test_step_outputs
        for output in outputs:
            all_mae += output["mae"]
            all_rmse += output["rmse"]
        test_mae = np.mean(all_mae)
        test_rmse = np.sqrt(np.mean(all_rmse))
        self.log('test_mae', test_mae)
        self.log('test_rmse', test_rmse)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            betas=(0.9, 0.95),
            weight_decay=self.args.weight_decay,
        )

        schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.33)
        return {"optimizer": optimizer, "lr_scheduler": schedular, "monitor": "val_mae"}

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # delete frozen clip parameters
        if not self.args.unfreeze_vit:
            for k in list(checkpoint["state_dict"].keys()):
                if k.startswith("models.clip") or k.startswith("models.img_encoder.clip") or k.startswith(
                        "models.text_encoder.clip") or k.startswith("models.img_encoder.vit"):
                    del checkpoint["state_dict"][k]

    def forward(self, img, prompt):
        """
        img: (1, 3, H, W)
        prompt: List[str]
        """
        return self.model(img, prompt)

    def overwrite_args(self, args):
        """Avoid the exception caused by lighting when loading incompatible args from models ckpt."""
        self.args = args
