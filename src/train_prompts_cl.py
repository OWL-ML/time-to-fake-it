import argparse
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as T
from loguru import logger

from dataset import IndxImageFolder
from models import PINAModel, SOYOModel, SPromptsModel
from trainers import PINATrainer, SOYOTrainer, SPromptsTrainer
from utils.constants import DEMO_ATTRS, METADATA_CSV, TASK_NAMES


def build_transforms():
    train_trsf = T.Compose(
        [
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=63 / 255),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    test_trsf = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return train_trsf, test_trsf


def build_task_datasets(dataset_dir):
    train_trsf, test_trsf = build_transforms()
    data_train, data_val, data_test, data_analysis = {}, {}, {}, {}

    for idx, task in enumerate(TASK_NAMES):
        print(f"Loading dataset for {task}...")
        common = dict(num_classes=2, return_mode="index")

        train_ds = IndxImageFolder(
            root=os.path.join(dataset_dir, task, "train"),
            transform=train_trsf,
            **common,
        )
        val_ds = IndxImageFolder(
            root=os.path.join(dataset_dir, task, "val"),
            transform=test_trsf,
            **common,
        )
        test_ds = IndxImageFolder(
            root=os.path.join(dataset_dir, task, "test"),
            transform=test_trsf,
            **common,
        )
        # Train data with test transforms — for post-task routing key building
        analysis_ds = IndxImageFolder(
            root=os.path.join(dataset_dir, task, "train"),
            transform=test_trsf,
            **common,
        )

        train_ds.attach_demographics(METADATA_CSV["train"], demo_attrs=DEMO_ATTRS)
        val_ds.attach_demographics(METADATA_CSV["val"], demo_attrs=DEMO_ATTRS)
        test_ds.attach_demographics(METADATA_CSV["test"], demo_attrs=DEMO_ATTRS)
        analysis_ds.attach_demographics(METADATA_CSV["train"], demo_attrs=DEMO_ATTRS)

        data_train[f"task_{idx}"] = train_ds
        data_val[f"task_{idx}"] = val_ds
        data_test[f"task_{idx}"] = test_ds
        data_analysis[f"task_{idx}"] = analysis_ds

    return data_train, data_val, data_test, data_analysis


def build_model_and_trainer(args, config, device, datasets):
    data_train, data_val, data_test, data_analysis = datasets
    # args dict for the reference model constructors
    model_args = {
        "dataset": "ai_face",
        "total_sessions": len(TASK_NAMES),
        # AI_FACE default: binary classification
        "class_num": config.get("class_num", 2),
        "prompt_length": args.prompt_length,
        "image_dim": 768,  # SOYO
        "hidden_dim": args.hidden_dim,  # SOYO
        "adapter_bottleneck": args.hidden_dim,  # PINA
        "ca_mode": args.ca_mode,  # PINA only
    }

    trainer_kwargs = dict(
        model_name=args.model_name,
        config=config,
        device=device,
        train_datasets=data_train,
        val_datasets=data_val,
        test_datasets=data_test,
        analysis_datasets=data_analysis,
    )

    if args.method == "soyo":
        model = SOYOModel(model_args).to(device)
        return SOYOTrainer(model=model, **trainer_kwargs)
    elif args.method == "sprompts":
        model = SPromptsModel(model_args).to(device)
        return SPromptsTrainer(model=model, **trainer_kwargs)
    elif args.method == "pina":
        model = PINAModel(model_args).to(device)
        return PINATrainer(model=model, **trainer_kwargs)
    else:
        raise ValueError(f"Unknown method: {args.method}")


def main(args):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using eval_only: {args.evaluate_only}")

    # Reproducibility — mirror train_cl.py
    base_seed = args.seed
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)
    random.seed(base_seed)
    cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)

    config = {
        "base_seed": base_seed,
        "class_num": 2,
        "total_sessions": len(TASK_NAMES),
        "soyo_dim": 768,
        "batch_size": args.batch_size,
        "num_workers": 4,
        # Resource profiling (PromptCLTrainer)
        "profile_resources": args.profile_resources,
        "profile_include_eval": args.profile_include_eval,
        # Task-0 vs later hyperparameters (reference convention)
        "init_lr": args.init_lr,
        "init_epochs": args.init_epochs,
        "init_weight_decay": args.weight_decay,
        "lr": args.lr,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "eta_min": 1e-6,
        # SOYO/PINA phase-2 (soyo_network) hyperparameters
        "soyo_epoch": args.soyo_epoch,
        "soyo_lr": args.soyo_lr,
        "soyo_weight_decay": args.soyo_weight_decay,
        "n_components": 2,
    }

    dataset_dir = os.environ.get("DATASET_DIR", "datasets/AI_FACE")
    print(f"Demographics: {METADATA_CSV}, columns {DEMO_ATTRS}")
    datasets = build_task_datasets(dataset_dir)

    trainer = build_model_and_trainer(args, config, device, datasets)
    print(f"Metrics will be saved to {trainer.metrics_file}")
    if args.evaluate_only:
        print(f"Evaluating saved checkpoints for {args.model_name} seed {args.seed}")

        trainer.evaluate_saved_checkpoints(
            predictions_dir="predictions_posthoc",
            metrics_file=f"metrics/{args.model_name}_seed_{args.seed}_posthoc_metrics.json",
        )
    else:
        for task_id in range(len(TASK_NAMES)):
            trainer.incremental_train(task_id)

        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed:.0f} seconds")
        trainer.save_metrics(time_to_train=elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", type=str, required=True, choices=["sprompts", "soyo", "pina"]
    )
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument(
        "--evaluate_only",
        type=lambda x: str(x).lower() in {"1", "true", "t", "yes", "y"},
        default=False,
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=777)

    # Prompt/adapter architecture
    parser.add_argument("--prompt_length", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument(
        "--ca_mode", type=str, default="deep", choices=["shallow", "deep"]
    )

    # Main-phase training
    parser.add_argument("--init_lr", type=float, default=1e-3)
    parser.add_argument("--init_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # SOYO/PINA phase-2
    parser.add_argument("--soyo_epoch", type=int, default=10)
    parser.add_argument("--soyo_lr", type=float, default=1e-3)
    parser.add_argument("--soyo_weight_decay", type=float, default=1e-4)

    # Resource profiling
    parser.add_argument(
        "--profile_resources",
        type=lambda x: str(x).lower() in {"1", "true", "t", "yes", "y"},
        default=True,
        help="Log per-task training time and memory usage (default: true)",
    )
    parser.add_argument(
        "--profile_include_eval",
        type=lambda x: str(x).lower() in {"1", "true", "t", "yes", "y"},
        default=False,
        help="Also log per-task evaluation time (default: false)",
    )

    args = parser.parse_args()
    if args.model_name is None:
        args.model_name = args.method
    main(args)
