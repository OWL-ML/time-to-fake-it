import argparse
import os
import random
import time

import numpy as np
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as T
from loguru import logger

from dataset.IndxImageFolder import IndxImageFolder
from trainers import (
    AccumulatedTrainer,
    DERPPTrainer,
    EWCTrainer,
    GradMatchTrainer,
)
from trainers.trainer import Trainer
from utils.constants import DEMO_ATTRS, METADATA_CSV, TASK_NAMES
from utils.prefetch_loader import create_dataloader
from utils.train_utils import train_and_evaluate

torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)


def train_model(args):
    logger.info(
        f"Starting training for model: {args.model_name} with batch size {args.batch_size} at start_task_idx={args.start_task}"
    )
    dataset_dir = os.environ.get("DATASET_DIR", "datasets/AI_FACE")
    start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # fix the seed for reproducibility
    seed = int(args.base_seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    num_classes = 2
    training_config = {
        "seed": seed,
        "base_seed": seed,
        "epochs": 25,
        "lr": 1e-5,
        "eta_min": 1e-6,
        "batch_size": int(args.batch_size),
        "num_workers": 4,
        "weight_decay": 1e-3,
        "num_classes": num_classes,
    }

    transform_train = T.Compose(
        [
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=63 / 255),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    transform_test = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    num_classes = 2

    data_train = {}
    data_val = {}
    data_test = {}
    accumulated_trainset = None
    accumulated_valset = None
    return_demo = args.derpp or args.grad_match
    for idx, generator_model in enumerate(TASK_NAMES):
        print(f"Loading dataset for {generator_model}...")
        trainset = IndxImageFolder(
            root=os.path.join(dataset_dir, generator_model, "train"),
            transform=transform_train,
            num_classes=2,
            return_demographics=return_demo,
        )

        valset = IndxImageFolder(
            root=os.path.join(dataset_dir, generator_model, "val"),
            transform=transform_test,
            num_classes=2,
            return_demographics=return_demo,
        )

        testset = IndxImageFolder(
            root=os.path.join(dataset_dir, generator_model, "test"),
            transform=transform_test,
            num_classes=2,
            return_demographics=True,
        )
        # Attach demographics to the trainset for logging buffer details for DERPP
        if return_demo:
            trainset.attach_demographics(METADATA_CSV["train"], demo_attrs=DEMO_ATTRS)
            valset.attach_demographics(METADATA_CSV["val"], demo_attrs=DEMO_ATTRS)
        # Attach demographics to the testset for saving probabilities and evaluations per demographic group
        testset.attach_demographics(METADATA_CSV["test"], demo_attrs=DEMO_ATTRS)
        if args.accumulated:
            accumulated_trainset = (
                trainset
                if accumulated_trainset is None
                else torch.utils.data.ConcatDataset([accumulated_trainset, trainset])
            )
            accumulated_valset = (
                valset
                if accumulated_valset is None
                else torch.utils.data.ConcatDataset([accumulated_valset, valset])
            )

        data_train[f"task_{idx}"] = create_dataloader(
            trainset if not args.accumulated else accumulated_trainset,
            training_config,
        )

        data_val[f"task_{idx}"] = create_dataloader(
            valset if not args.accumulated else accumulated_valset,
            training_config,
            is_trains=False,
        )
        # test set is never accumulated
        data_test[f"task_{idx}"] = create_dataloader(
            testset,
            training_config,
            is_trains=False,
        )
    model = timm.create_model(
        "vit_base_patch16_224", pretrained=True, num_classes=num_classes
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    if args.ewc:
        training_config.update({"is_ewc": True, "lambda_ewc": args.lambda_ewc})
        print("Initializing EWC trainer")
        trainer = EWCTrainer(model, args.model_name, training_config, device, criterion)
    elif args.derpp:
        training_config.update(
            {
                "use_replay": True,
                "alpha": args.alpha,
                "beta": args.beta,
                "capacity": args.capacity,
                "replay_size": args.replay_size,
            }
        )
        print("Initializing DERPP trainer")
        trainer = DERPPTrainer(
            model, args.model_name, training_config, device, criterion
        )
    elif args.grad_match:
        training_config.update(
            {
                "use_replay": True,
                "use_coreset": True,
                "beta": args.beta,
                "capacity": args.capacity,
                "replay_size": args.replay_size,
            }
        )
        print("Initializing GradMatch trainer")
        trainer = GradMatchTrainer(
            model, args.model_name, training_config, device, criterion
        )
    elif args.accumulated:
        print(f"Initializing accumulated trainer for {args.model_name}")
        trainer = AccumulatedTrainer(
            model=model,
            model_name=args.model_name,
            config=training_config,
            device=device,
            criterion=criterion,
        )
    else:
        print(f"Initializing naive sequential trainer")
        trainer = Trainer(model, args.model_name, training_config, device, criterion)
        print(f"Metrics will be saved to {trainer.metrics_file}")
    train_and_evaluate(
        trainer, data_train, data_val, data_test, start_task_idx=args.start_task
    )

    end_time = time.time()
    print(f"Training completed in {end_time - start_time} seconds")
    trainer.save_metrics(time_to_train=(end_time - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--base_seed", type=int, default=777)
    parser.add_argument("--accumulated", default=False, type=bool)
    parser.add_argument("--ewc", default=False, type=bool)
    parser.add_argument("--grad_match", default=False, type=bool)
    parser.add_argument("--model_name", default="naive", type=str)
    parser.add_argument("--start_task", default=0, type=int)
    parser.add_argument("--lambda_ewc", default=1, type=int)
    parser.add_argument("--derpp", default=False, type=bool)

    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--replay_size", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=1000)
    parser.add_argument("--use_coreset", default=False, type=bool)

    args = parser.parse_args()
    train_model(args)
