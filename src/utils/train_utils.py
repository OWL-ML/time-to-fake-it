import os

import numpy as np
from loguru import logger

import wandb


def train_and_evaluate(trainer, data_train, data_val, data_test, start_task_idx=0):
    """Train and evaluate the model across all tasks."""
    run = wandb.init(project="ai_face_detection", config=trainer.config)

    uses_replay = hasattr(trainer, "buffer") and trainer.config.get("use_replay", False)
    n_tasks = len(data_train)

    # ------------------------------------------------------------------
    # Resolve the resume point BEFORE touching any state.
    # ------------------------------------------------------------------
    if start_task_idx == 0:
        start_task_idx = trainer.get_start_task()
    print(f"Resolved start_task_idx: {start_task_idx}")
    if start_task_idx > 0:
        resume_model_path = os.path.join(
            trainer.checkpoint_path, f"{trainer.model_name}_{start_task_idx}.pt"
        )
        logger.info(f"Resuming from task {start_task_idx} at {resume_model_path}")

        if not os.path.exists(resume_model_path):
            logger.warning(
                "No model checkpoint at {}; restarting from task 1.", resume_model_path
            )
            start_task_idx = 0

    buffer_restored = False
    logger.info(
        f"Resuming from task {start_task_idx} as {hasattr(trainer, 'load_buffer')} and {uses_replay}"
    )
    if uses_replay and start_task_idx > 0 and hasattr(trainer, "load_buffer"):
        buffer_path = trainer.buffer_checkpoint_path()
        if buffer_path.exists():
            last_completed = trainer.load_buffer()
            buffer_restored = True
            logger.info(
                f"Buffer restored from {buffer_path} at last_completed={last_completed}"
            )
            if last_completed != start_task_idx - 1:
                logger.warning(
                    "Buffer snapshot is from the end of task {} but the model resumes "
                    "at task {} — buffer and model are out of step.",
                    last_completed + 1,
                    start_task_idx + 1,
                )
        else:
            logger.warning(
                "No buffer snapshot at {}; replay state cannot be resumed. "
                "Restarting from task 1.",
                buffer_path,
            )
            start_task_idx = 0

    logger.info(
        "Starting at task {} (buffer_restored={})", start_task_idx + 1, buffer_restored
    )

    # ------------------------------------------------------------------
    for task_idx, (train_loader, val_loader) in enumerate(
        zip(data_train.values(), data_val.values())
    ):
        trainer.current_task = task_idx
        if hasattr(trainer, "begin_stage"):
            trainer.begin_stage(task_idx)

        model_path = os.path.join(
            trainer.checkpoint_path, f"{trainer.model_name}_{task_idx + 1}.pt"
        )
        train_on_current_task = True
        test_results_available = False

        # ---------------- task 0: seed from the naive checkpoint ----------------
        if task_idx == 0:
            naive_model_checkpoint = os.path.join(
                "checkpoints/naive",
                f"seed_run_{trainer.config.get('base_seed', 42)}",
                "naive_1.pt",
            )
            if os.path.exists(naive_model_checkpoint):
                trainer.load_checkpoint(naive_model_checkpoint)

                if getattr(trainer, "is_ewc", False) and trainer.lambda_ewc is not None:
                    if trainer.load_ewc_state():
                        continue
                    logger.warning("EWC state could not be loaded")

                trainer.reset_optimizer()

                if trainer.config.get("use_replay", False):
                    if buffer_restored:
                        logger.info(
                            "Buffer restored from snapshot ({} samples); skipping the "
                            "task 1 preload.",
                            len(trainer.buffer),
                        )
                        continue

                    trainer.buffer.set_task_id(task_idx + 1)
                    logger.info(
                        "Pre-loading samples from task {} into buffer", task_idx + 1
                    )
                    if trainer.model_name == "derpp":
                        trainer.preload_task_into_buffer(train_loader)
                    elif trainer.model_name == "grad_match" and trainer.config.get(
                        "use_coreset", False
                    ):
                        images, targets, image_paths, demographics = (
                            trainer.select_coreset(train_loader, val_loader)
                        )
                        trainer.buffer.add(
                            examples=images.detach().cpu(),
                            labels=targets.detach().cpu(),
                            image_paths=image_paths,
                            metas=demographics,
                        )
                    logger.info(
                        "{} samples loaded into the buffer.", len(trainer.buffer)
                    )

                    # Task 0 never reaches trainer.train(), so snapshot here.
                    if hasattr(trainer, "save_buffer"):
                        trainer.save_buffer()
                    continue

                train_on_current_task = False

        # ---------------- already-completed tasks ----------------
        if start_task_idx > 0 and task_idx < start_task_idx:
            if os.path.exists(model_path):
                logger.info("Skipping training on task {}", task_idx + 1)
                train_on_current_task = False
                test_results_available = trainer.check_test_results()
                # Load unconditionally: this leaves the model at the end-state of
                # the last completed task, which is what task `start_task_idx`
                # must train from.
                trainer.load_checkpoint(model_path)
            else:
                logger.warning(
                    "Checkpoint not found for task {}; training it from scratch.",
                    task_idx + 1,
                )

        # ---------------- train ----------------
        if train_on_current_task:
            if uses_replay:
                trainer.buffer.set_task_id(task_idx + 1)
            trainer.reset_optimizer()
            logger.info(
                "Starting training for task {} with learning rate {}",
                task_idx + 1,
                trainer.optimizer.param_groups[0]["lr"],
            )
            trainer.train(train_loader, val_loader, run)
            trainer.save_checkpoint()
            if uses_replay and hasattr(trainer, "save_buffer"):
                trainer.save_buffer()
            trainer.reset_early_stopping()

        # ---------------- test ----------------
        if not test_results_available:
            logger.info("Testing model trained through task {}", task_idx + 1)
            task_accuracies = trainer.test_all_tasks(data_test)
            avg_acc = np.mean(task_accuracies[: trainer.current_task + 1])
            logger.info("Average accuracy after task {}: {:.4f}", task_idx + 1, avg_acc)

            metrics_dict = {"task_id": task_idx + 1, "average_accuracy": avg_acc}
            if run:
                run.log(metrics_dict)
            for i, acc in enumerate(task_accuracies):
                metrics_dict[f"task_{i + 1}_acc_after_task_{task_idx + 1}"] = acc
            trainer.task_metrics.update({f"{task_idx + 1}": metrics_dict})

        # ---------------- EWC Fisher ----------------
        if (
            getattr(trainer, "is_ewc", False)
            and trainer.lambda_ewc is not None
            and task_idx < n_tasks - 1
        ):
            logger.info("Computing Fisher information for task {}...", task_idx + 1)
            trainer.compute_fisher_and_store(train_loader)
            logger.info("Penalty centers: {}", len(trainer.penalty_centers))
            if trainer.penalty_centers:
                test_penalty = trainer.penalty()
                logger.info(
                    "Penalty raw {:.6f} | scaled {:.2f}",
                    test_penalty.item(),
                    (trainer.lambda_ewc * test_penalty).item(),
                )

        if hasattr(trainer, "end_stage"):
            trainer.end_stage(task_idx)

    if uses_replay:
        trainer.buffer.finalize()
    trainer.save_metrics()
    wandb.finish()
