import argparse
import random
from pathlib import Path

import PIL
import numpy as np
import torch

from torch.utils.data import RandomSampler, DataLoader

import util.misc as misc
from models.model import Model
from util.CARPK import CARPK
from util.FSC147_2 import FSC147
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything

from util.ShanghaiTech import ShanghaiTech
from util.constant import SCALE_FACTOR

import cv2
import gradio as gr
import torchvision.transforms.functional as TF


def get_args_parser():
    parser = argparse.ArgumentParser('CLIP-Count', add_help=False)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "app"],
                        help="train or test or an interactive application")
    parser.add_argument("--exp_name", type=str, default="exp", help="experiment name")
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--backbone', default="b14", choices=["b16", "b14", "l14"],
                        type=str, help="backbone of clip")
    parser.add_argument('--decoder_depth', default=4, type=int, help='Number of FIM layers')
    parser.add_argument('--decoder_head', default=8, type=int, help='Number of attention heads for FIM')

    parser.add_argument('--use_mixed_fim', default=True, type=misc.str2bool,
                        help="whether to use hierarchical patch-text interaction")
    parser.add_argument('--unfreeze_vit', default=False, type=misc.str2bool,
                        help="whether to unfreeze CLIP vit i.e., finetune CLIP")
    parser.add_argument('--use_fim', default=False, type=misc.str2bool, help="whether to use naive interaction")

    # contrastive loss related
    parser.add_argument('--use_coop', default=False, type=misc.str2bool,
                        help='whether to perform context learning for text prompts.')
    parser.add_argument('--coop_width', default=4, type=int, help="width of context (how many token to be learned)")
    parser.add_argument('--coop_require_grad', default=True, type=misc.str2bool,
                        help="whether to require grad for context learning")
    parser.add_argument('--use_vpt', default=True, type=misc.str2bool,
                        help='whether to perform visual prompt learning.')
    parser.add_argument('--vpt_width', default=40, type=int, help="width of visual prompt (how many token each layer)")
    parser.add_argument('--vpt_depth', default=10, type=int, help="depth of visual prompt (how many layer)")

    parser.add_argument("--use_contrast", default=False, type=misc.str2bool, help="whether to use contrasitive loss")
    parser.add_argument("--w_contrast", default=1.0, type=float, help="weight of contrastive loss")
    parser.add_argument("--noise_text_ratio", default=0.0, type=float, help="ratio of noise text")
    parser.add_argument('--normalize_contrast', default=True, type=misc.str2bool,
                        help="whether to normalize contrastive loss")
    parser.add_argument('--contrast_pos', default="pre", choices=["pre", "post"], type=str,
                        help="Use contrastive loss before or after the interaction")
    parser.add_argument('--contrast_pre_epoch', default=20, type=int,
                        help="how many epoch to use contrastive pretraining")

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/', type=str,
                        help='dataset path')
    parser.add_argument('--dataset_type', default="FSC", type=str,
                        choices=["FSC", "CARPK", "COCO", "ShanghaiTech"])

    parser.add_argument('--output_dir', default='./out',
                        help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--ckpt', default=None,
                        type=str,
                        help='path of resume from checkpoint')
    parser.add_argument('--version', default=12, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # log related
    parser.add_argument('--log_dir', default='./out',
                        help='path where to tensorboard log')
    parser.add_argument('--log_test_img', default=True, type=bool,
                        help="whehter to log overlaied density map when validation and testing.")
    parser.add_argument('--dont_log', action='store_true', help='do not log to tensorboard')
    parser.add_argument('--val_freq', default=1, type=int, help='check validation every val_freq epochs')

    # log setup
    parser.add_argument('--exp_note', default="", type=str, help="experiment note")
    return parser


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    seed = args.seed
    seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.set_float32_matmul_precision('high')

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    dataset_train = FSC147(split="train")
    sampler_train = RandomSampler(dataset_train)

    train_dataloader = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    # the val set for training.
    dataset_val = FSC147(split="val")
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    save_callback = pl.callbacks.ModelCheckpoint(monitor='val_mae', save_top_k=4, mode='min',
                                                 filename='{epoch}-{val_mae:.2f}')

    val_dataloader = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    model = Model(args)

    logger = pl.loggers.TensorBoardLogger("lightning_logs", name=args.exp_name, version=args.version)
    trainer = Trainer(
        accelerator="gpu",
        callbacks=[save_callback],
        accumulate_grad_batches=args.accum_iter,
        precision="16-mixed",
        max_epochs=args.epochs,
        logger=logger,
        check_val_every_n_epoch=args.val_freq,
    )
    if args.mode == "train":
        if args.ckpt is not None:
            trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=args.ckpt)
        else:
            trainer.fit(model, train_dataloader, val_dataloader)
    elif args.mode == "test":
        if args.dataset_type == "FSC":
            dataset_val = FSC147(split="val")
            dataset_test = FSC147(split="test")
        elif args.dataset_type == "COCO":
            dataset_val = FSC147(split="val_coco", resize_val=False)
            dataset_test = FSC147(split="test_coco")
        elif args.dataset_type == "CARPK":
            dataset_val = dataset_test = CARPK(None, split="test")
        elif args.dataset_type == "ShanghaiTech":
            dataset_val = dataset_test = ShanghaiTech(None, split="test", part="B")

        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        # when inference, batch size is always 1
        val_dataloader = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        test_dataloader = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        if args.ckpt is None:
            raise ValueError("Please specify a checkpoint to test")
        model = Model.load_from_checkpoint(args.ckpt, strict=False)
        model.overwrite_args(args)
        model.eval()
        if args.dataset_type == "FSC" or args.dataset_type == "COCO":  # CARPK and ShanghaiTech do not have val set
            print("====Metric on val set====")
            trainer.test(model, val_dataloader)
        print("====Metric on test set====")
        trainer.test(model, test_dataloader)

    elif args.mode == "app":
        if args.ckpt is None:
            raise ValueError("Please specify a checkpoint to test")
        model = Model.load_from_checkpoint(args.ckpt, strict=False)
        model.eval()


        def infer(img, prompt):
            model.eval()
            model.model = model.model.cuda()
            with torch.no_grad():
                # reshape height to 384, keep aspect ratio
                img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
                img = TF.resize(img, [384, 384])

                img = img.float() / 255.
                img = torch.clamp(img, 0, 1)
                prompt = [prompt]
                with torch.cuda.amp.autocast():
                    raw_h, raw_w = img.shape[2:]
                    patches, _ = misc.sliding_window(img, stride=128)
                    # covert to batch
                    patches = torch.from_numpy(patches).float().to(img.device)
                    prompt = np.repeat(prompt, patches.shape[0], axis=0)
                    output, extra_out = model.forward(patches, prompt)
                    output.unsqueeze_(1)
                    output = misc.window_composite(output, stride=128)
                    output = output.squeeze(1)
                    # crop to original width
                    output = output[:, :, :raw_w]
                pred_cnt = torch.sum(output[0] / SCALE_FACTOR).item()
                pred_density = output[0].detach().cpu().numpy()
                # normalize
                pred_density = pred_density / pred_density.max()
                pred_density_write = 1. - pred_density
                pred_density_write = cv2.applyColorMap(np.uint8(255 * pred_density_write), cv2.COLORMAP_JET)
                pred_density_write = pred_density_write / 255.
                pred_log_rgb = cv2.applyColorMap(np.uint8(255 * pred_density), cv2.COLORMAP_JET)
                im = PIL.Image.fromarray(pred_log_rgb)
                im.save('pred_log_rgb.png')

                img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

                heatmap_pred = 0.33 * img + 0.67 * pred_density_write
                heatmap_pred = heatmap_pred / heatmap_pred.max()

                sim_map = extra_out['sim_map_16x'][0].detach().cpu().numpy()
                sim_map = (sim_map - np.min(sim_map)) / (np.max(sim_map) - np.min(sim_map))
                sim_map_rgb = cv2.applyColorMap(np.uint8(255 * sim_map[0, :, :]), cv2.COLORMAP_JET)
                im = PIL.Image.fromarray(sim_map_rgb.astype(np.uint8))
                im.save('sim_map_rgb.png')
            return heatmap_pred, pred_cnt


        demo = gr.Interface(
            fn=infer,
            inputs=[
                # height = 384, keep aspect ratio
                gr.components.Image(label="Image"),
                gr.components.Textbox(lines=1, label="Prompt (What would you like to count)"),
            ],
            outputs=["image", "number"],
            title="Dino-Count",
            description="A unified counting model to count them all.",
        )
        demo.launch(share=False)
