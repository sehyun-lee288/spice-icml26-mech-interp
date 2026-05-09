# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

import dinov3.distributed as distributed
from dinov3.checkpointer import (
    CheckpointRetentionPolicy,
    cleanup_checkpoint,
    find_latest_checkpoint,
    keep_last_n_checkpoints,
)
from dinov3.data import SamplerType, make_data_loader, make_dataset
from dinov3.data.adapters import DatasetWithEnumeratedTargets
from dinov3.data.transforms import (
    CROP_DEFAULT_SIZE,
    RESIZE_DEFAULT_SIZE,
    make_classification_eval_transform,
    make_classification_train_transform,
)
from dinov3.eval.data import create_train_dataset_dict, get_num_classes, pad_multilabel_and_collate
from dinov3.eval.metrics import ClassificationMetricType, build_classification_metric
from dinov3.eval.setup import ModelConfig, load_model_and_context
from dinov3.eval.utils import LossType, ModelWithIntermediateLayers, average_metrics, evaluate
from dinov3.eval.utils import save_results as default_save_results_func
from dinov3.logging import MetricLogger, SmoothedValue

logger = logging.getLogger("dinov3")

RESULTS_FILENAME = "results-linear.csv"
# Can be several keys, depending on if multiple test sets are chosen and if doing few-shot
MAIN_METRICS = [".*_accuracy(_mean)?"]


class OptimizerType(Enum):
    SGD = "sgd"
    ADAMW = "adamw"

    def get_optimizer(self, optim_param_groups):
        if self == OptimizerType.ADAMW:
            optimizer = torch.optim.AdamW(optim_param_groups, weight_decay=0)
        else:
            optimizer = torch.optim.SGD(optim_param_groups, momentum=0.9, weight_decay=0)
        return optimizer


class SchedulerType(Enum):
    COSINE_ANNEALING = "cosine_annealing"
    ONE_CYCLE = "one_cycle"

    def get_scheduler(self, optimizer, optim_param_groups, epoch_length, epochs, max_iter):
        if self == SchedulerType.ONE_CYCLE:
            lr_list = [optim_param_groups[i]["lr"] for i in range(len(optim_param_groups))]
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr_list, steps_per_epoch=epoch_length, epochs=epochs
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)
        return scheduler


def _write_results_file(results_dict: Dict[str, Any], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, RESULTS_FILENAME)
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results_dict, handle, indent=2)
    logger.info("Saved results to %s", results_path)
    return results_path


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run linear evaluation for a frozen backbone.")
    parser.add_argument("--config-file", required=False, help="Path to the model config file.")
    parser.add_argument("--pretrained-weights", default=None, help="Path to pretrained weights.")
    parser.add_argument("--dino-hub", default=None, help="Name of the torch.hub entry for loading a model.")

    parser.add_argument("--train-dataset", required=True, help="Training dataset spec string.")
    parser.add_argument("--val-dataset", required=True, help="Validation dataset spec string.")
    parser.add_argument("--test-datasets", nargs="*", default=[], help="Optional extra test datasets.")
    parser.add_argument(
        "--test-metric-types",
        nargs="*",
        default=[],
        help="Optional metric types for test datasets (defaults to validation metric type).",
    )

    parser.add_argument("--output-dir", required=True, help="Directory to store checkpoints/results.")
    parser.add_argument("--save-results", action="store_true", help="Save predictions for the best classifier.")

    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size per GPU.")
    parser.add_argument("--val-batch-size", type=int, default=256, help="Eval batch size per GPU.")
    parser.add_argument("--num-workers", type=int, default=8, help="Training dataloader workers.")
    parser.add_argument("--val-num-workers", type=int, default=8, help="Evaluation dataloader workers.")

    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for the linear head.")
    parser.add_argument("--n-last-blocks", type=int, default=1, help="Number of last backbone blocks to use.")
    parser.add_argument(
        "--loss-type",
        choices=[loss_type.value for loss_type in LossType],
        default=LossType.CROSS_ENTROPY.value,
        help="Loss to train the linear head.",
    )
    parser.add_argument(
        "--optimizer",
        choices=[opt.value for opt in OptimizerType],
        default=OptimizerType.SGD.value,
        help="Optimizer to train the linear head.",
    )
    parser.add_argument(
        "--scheduler",
        choices=[scheduler.value for scheduler in SchedulerType],
        default=SchedulerType.COSINE_ANNEALING.value,
        help="Learning-rate scheduler to use.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--epoch-length", type=int, default=1250, help="Number of iterations per epoch.")
    parser.add_argument(
        "--save-checkpoint-iterations",
        type=int,
        default=0,
        help="Iterations between checkpoints (0 = one per epoch).",
    )
    parser.add_argument(
        "--eval-period-iterations", type=int, default=0, help="Iterations between eval runs (0 = one per epoch)."
    )
    parser.add_argument(
        "--checkpoint-retention",
        choices=[policy.value for policy in CheckpointRetentionPolicy],
        default=CheckpointRetentionPolicy.NONE.value,
        help="Which checkpoints to keep after cleanup.",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true", help="Resume if checkpoint exists.")
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Start from scratch even if checkpoints are present.",
    )
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--classifier-fpath",
        default=None,
        help="Path containing pretrained linear classifiers to resume from (optional).",
    )

    parser.add_argument("--resize-size", type=int, default=RESIZE_DEFAULT_SIZE)
    parser.add_argument("--crop-size", type=int, default=CROP_DEFAULT_SIZE)

    parser.add_argument("--few-shot", action="store_true", help="Enable few-shot evaluation trick.")
    parser.add_argument("--few-shot-k", type=float, default=None, help="Few-shot k or percent value.")
    parser.add_argument("--few-shot-tries", type=int, default=1, help="Number of few-shot trials.")

    parser.add_argument(
        "--val-metric-type",
        choices=[metric.value for metric in ClassificationMetricType],
        default=ClassificationMetricType.MEAN_ACCURACY.value,
    )
    parser.add_argument("--metrics-file-name", default="results_eval_linear.json")

    args = parser.parse_args(argv)
    return args


def _build_runtime_config(args: argparse.Namespace):
    if not any([args.config_file, args.dino_hub]):
        raise ValueError("You must provide either --config-file or --dino-hub to load a model.")

    train_config = SimpleNamespace(
        dataset=args.train_dataset,
        val_dataset=args.val_dataset,
        val_metric_type=ClassificationMetricType(args.val_metric_type),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        n_last_blocks=args.n_last_blocks,
        loss_type=LossType(args.loss_type),
        optimizer_type=OptimizerType(args.optimizer),
        scheduler_type=SchedulerType(args.scheduler),
        epochs=args.epochs,
        epoch_length=args.epoch_length,
        save_checkpoint_iterations=args.save_checkpoint_iterations or None,
        eval_period_iterations=args.eval_period_iterations or None,
        checkpoint_retention_policy=CheckpointRetentionPolicy(args.checkpoint_retention),
        resume=args.resume,
        classifier_fpath=args.classifier_fpath,
    )

    eval_config = SimpleNamespace(
        test_datasets=tuple(args.test_datasets),
        test_metric_types=tuple(ClassificationMetricType(metric) for metric in args.test_metric_types),
        batch_size=args.val_batch_size,
        num_workers=args.val_num_workers,
    )

    transform_config = SimpleNamespace(resize_size=args.resize_size, crop_size=args.crop_size)
    few_shot_config = SimpleNamespace(enable=args.few_shot, k_or_percent=args.few_shot_k, n_tries=args.few_shot_tries)

    config = SimpleNamespace(
        model=ModelConfig(
            config_file=args.config_file,
            pretrained_weights=args.pretrained_weights,
            dino_hub=args.dino_hub,
        ),
        train=train_config,
        eval=eval_config,
        transform=transform_config,
        few_shot=few_shot_config,
        save_results=args.save_results,
        output_dir=args.output_dir,
        metrics_file_name=args.metrics_file_name,
    )
    return config


def has_ddp_wrapper(m: nn.Module) -> bool:
    return isinstance(m, DistributedDataParallel)


def remove_ddp_wrapper(m: nn.Module) -> nn.Module:
    return m.module if has_ddp_wrapper(m) else m


def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, out_dim, use_n_blocks, use_avgpool, num_classes=1000):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear(output)


class AllClassifiers(nn.Module):
    def __init__(self, classifiers_dict):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs):
        return {k: v.forward(inputs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)


class LinearPostprocessor(nn.Module):
    def __init__(self, linear_classifier, class_mapping=None):
        super().__init__()
        self.linear_classifier = linear_classifier
        self.register_buffer("class_mapping", None if class_mapping is None else torch.LongTensor(class_mapping))

    def forward(self, samples, targets):
        preds = self.linear_classifier(samples)
        return {
            "preds": preds[:, self.class_mapping] if self.class_mapping is not None else preds,
            "target": targets,
        }


def scale_lr(learning_rate, batch_size):
    return learning_rate * (batch_size * distributed.get_world_size()) / 256.0


def setup_linear_classifiers(sample_output, n_last_blocks, learning_rate, batch_size, num_classes=1000):
    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []
    avgpool = True
    lr = scale_lr(learning_rate, batch_size)
    out_dim = create_linear_input(sample_output, use_n_blocks=n_last_blocks, use_avgpool=avgpool).shape[1]
    linear_classifier = LinearClassifier(out_dim, use_n_blocks=n_last_blocks, use_avgpool=avgpool, num_classes=num_classes)
    linear_classifier = linear_classifier.cuda()
    classifier_name = f"classifier_{n_last_blocks}_blocks_avgpool_{avgpool}_lr_{lr:.5f}".replace(".", "_")
    linear_classifiers_dict[classifier_name] = linear_classifier
    optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    linear_classifiers = AllClassifiers(linear_classifiers_dict)
    if distributed.is_enabled():
        linear_classifiers = nn.parallel.DistributedDataParallel(linear_classifiers)

    return linear_classifiers, optim_param_groups


def make_eval_transform(config):
    if config.resize_size / config.crop_size != 256 / 224:
        logger.warning(
            f"Default resize / crop ratio is 256 / 224, here we have {config.resize_size} / {config.crop_size}"
        )
    transform = make_classification_eval_transform(resize_size=config.resize_size, crop_size=config.crop_size)
    return transform


def make_eval_data_loader(
    *,
    test_dataset_str,
    transform_config,
    batch_size,
    num_workers,
    metric_type,
):
    transform = make_eval_transform(transform_config)
    test_dataset = make_dataset(dataset_str=test_dataset_str, transform=transform)

    class_mapping = None
    if hasattr(test_dataset, "get_imagenet_class_mapping"):
        class_mapping = test_dataset.get_imagenet_class_mapping()

    test_data_loader = make_data_loader(
        dataset=DatasetWithEnumeratedTargets(test_dataset, pad_dataset=True, num_replicas=distributed.get_world_size()),
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
        persistent_workers=False,
        collate_fn=pad_multilabel_and_collate if metric_type == ClassificationMetricType.ANY_MATCH_ACCURACY else None,
    )
    return test_data_loader, class_mapping


@dataclass
class Evaluator:
    batch_size: int
    num_workers: int
    transform_config: Any
    dataset_str: str
    metric_type: ClassificationMetricType
    metrics_file_path: str
    training_num_classes: int
    save_results_func: Optional[Callable]

    def __post_init__(self):
        self.data_loader, self.class_mapping = make_eval_data_loader(
            test_dataset_str=self.dataset_str,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            transform_config=self.transform_config,
            metric_type=self.metric_type,
        )
        self.main_metric_name = f"{self.dataset_str}_accuracy"

    @torch.no_grad()
    def _evaluate_linear_classifiers(
        self,
        *,
        feature_model,
        linear_classifiers,
        iteration,
        prefixstring="",
        best_classifier_on_val=None,
        accumulate_results=False,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, torch.Tensor]]]:
        logger.info("running validation !")

        num_classes = len(self.class_mapping) if self.class_mapping is not None else self.training_num_classes
        metric = build_classification_metric(self.metric_type, num_classes=num_classes)
        postprocessors = {
            k: LinearPostprocessor(v, self.class_mapping) for k, v in linear_classifiers.classifiers_dict.items()
        }
        metrics = {k: metric.clone() for k in linear_classifiers.classifiers_dict}

        _, results_dict_temp, accumulated_results = evaluate(
            feature_model,
            self.data_loader,
            postprocessors,
            metrics,
            torch.cuda.current_device(),
            accumulate_results=accumulate_results,
        )

        logger.info("")
        results_dict = {}
        max_accuracy = 0
        best_classifier = ""
        for _, (classifier_string, metric) in enumerate(results_dict_temp.items()):
            logger.info(f"{prefixstring} -- Classifier: {classifier_string} * {metric}")
            if (
                best_classifier_on_val is None and metric["top-1"].item() > max_accuracy
            ) or classifier_string == best_classifier_on_val:
                max_accuracy = metric["top-1"].item()
                best_classifier = classifier_string

        results_dict["best_classifier"] = {"name": best_classifier, "accuracy": max_accuracy}

        logger.info(f"best classifier: {results_dict['best_classifier']}")

        accumulated_best_results = None
        if accumulated_results is not None:
            accumulated_best_results = accumulated_results[best_classifier]

        if distributed.is_main_process():
            with open(self.metrics_file_path, "a") as f:
                f.write(f"iter: {iteration}\n")
                for k, v in results_dict.items():
                    f.write(json.dumps({k: v}) + "\n")
                f.write("\n")

        return results_dict, accumulated_best_results

    def evaluate_and_maybe_save(
        self,
        feature_model,
        linear_classifiers,
        iteration: int,
        best_classifier_on_val: Optional[Any] = None,
        save_filename_suffix: str = "",
        prefixstring: str = "",
    ):
        logger.info(f"Testing on {self.dataset_str}")
        save_results = self.save_results_func is not None
        full_results_dict, accumulated_best_results = self._evaluate_linear_classifiers(
            feature_model=feature_model,
            linear_classifiers=remove_ddp_wrapper(linear_classifiers),
            iteration=iteration,
            prefixstring=prefixstring,
            best_classifier_on_val=best_classifier_on_val,
            accumulate_results=save_results,
        )
        if self.save_results_func is not None:
            self.save_results_func(
                filename_suffix=f"{self.dataset_str}{save_filename_suffix}", **accumulated_best_results
            )

        results_dict = {
            self.main_metric_name: 100.0 * full_results_dict["best_classifier"]["accuracy"],
            "best_classifier": full_results_dict["best_classifier"]["name"],
        }
        return results_dict


def make_evaluators(
    eval_config,
    val_metric_type: ClassificationMetricType,
    val_dataset: str,
    transform_config,
    metrics_file_path: str,
    training_num_classes: int,
    save_results_func: Optional[Callable],
):
    test_metric_types = eval_config.test_metric_types
    if len(test_metric_types) == 0:
        test_metric_types = (val_metric_type,) * len(eval_config.test_datasets)
    else:
        assert len(test_metric_types) == len(eval_config.test_datasets)
    val_evaluator, *test_evaluators = [
        Evaluator(
            dataset_str=dataset_str,
            batch_size=eval_config.batch_size,
            num_workers=eval_config.num_workers,
            transform_config=transform_config,
            metric_type=metric_type,
            metrics_file_path=metrics_file_path,
            training_num_classes=training_num_classes,
            save_results_func=save_results_func,
        )
        for dataset_str, metric_type in zip(
            (val_dataset,) + tuple(eval_config.test_datasets),
            (val_metric_type,) + tuple(test_metric_types),
        )
    ]
    return val_evaluator, test_evaluators


def setup_linear_training(
    *,
    config,
    sample_output: torch.Tensor,
    training_num_classes: int,
    checkpoint_output_dir: str,
):
    linear_classifiers, optim_param_groups = setup_linear_classifiers(
        sample_output,
        config.n_last_blocks,
        config.learning_rate,
        config.batch_size,
        training_num_classes,
    )
    max_iter = config.epochs * config.epoch_length
    optimizer = config.optimizer_type.get_optimizer(optim_param_groups=optim_param_groups)
    scheduler = config.scheduler_type.get_scheduler(
        optimizer=optimizer,
        optim_param_groups=optim_param_groups,
        epoch_length=config.epoch_length,
        epochs=config.epochs,
        max_iter=max_iter,
    )
    start_iter = 0
    best_accuracy = -1
    if config.resume and (
        last_checkpoint_dir := find_latest_checkpoint(config.classifier_fpath or checkpoint_output_dir)
    ):
        logger.info(f"Checkpoint found {last_checkpoint_dir}")
        checkpoint = torch.load(last_checkpoint_dir / "checkpoint.pth")
        start_iter = checkpoint.get("iteration", -1) + 1
        best_accuracy = checkpoint.get("best_accuracy", -1)
        linear_classifiers.load_state_dict(checkpoint["linear_classifiers"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    if config.loss_type == LossType.BINARY_CROSS_ENTROPY:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    return (
        linear_classifiers,
        start_iter,
        max_iter,
        criterion,
        optimizer,
        scheduler,
        best_accuracy,
    )


def train_linear_classifiers(
    *,
    feature_model,
    train_dataset,
    train_config,
    training_num_classes: int,
    val_evaluator: Evaluator,
    checkpoint_output_dir: str,
):
    (linear_classifiers, start_iter, max_iter, criterion, optimizer, scheduler, best_accuracy,) = setup_linear_training(
        config=train_config,
        sample_output=feature_model(train_dataset[0][0].unsqueeze(0).cuda()),
        training_num_classes=training_num_classes,
        checkpoint_output_dir=checkpoint_output_dir,
    )
    checkpoint_period = train_config.save_checkpoint_iterations or train_config.epoch_length
    eval_period = train_config.eval_period_iterations or train_config.epoch_length

    sampler_type = SamplerType.INFINITE
    train_data_loader = make_data_loader(
        dataset=train_dataset,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        shuffle=True,
        seed=0,
        sampler_type=sampler_type,
        sampler_advance=start_iter,
        drop_last=True,
        persistent_workers=True,
    )

    iteration = start_iter
    logger.info("Starting training from iteration {}".format(start_iter))
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6g}"))
    header = "Training"
    for data, labels in metric_logger.log_every(
        train_data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):
        data = data.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        features = feature_model(data)
        outputs = linear_classifiers(features)

        if len(labels.shape) > 1:
            labels = labels.float()
        losses = {f"loss_{k}": criterion(v, labels) for k, v in outputs.items()}
        loss = sum(losses.values())

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()
        scheduler.step()

        # log
        if iteration % 10 == 0:
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # Checkpointing
        is_last_iteration = (iteration + 1) == max_iter
        is_ckpt_iteration = ((iteration + 1) % checkpoint_period == 0) or is_last_iteration
        if is_ckpt_iteration:
            ckpt_dir = Path(checkpoint_output_dir).expanduser()
            if distributed.is_subgroup_main_process():
                ckpt_sub_dir = "final" if is_last_iteration else str(iteration)
                (ckpt_dir / ckpt_sub_dir).mkdir(parents=True, exist_ok=True)
                checkpoint = {
                    "iteration": iteration,
                    "linear_classifiers": linear_classifiers.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_accuracy": best_accuracy,
                }
                torch.save(checkpoint, ckpt_dir / ckpt_sub_dir / "checkpoint.pth")
                keep_last_n_checkpoints(ckpt_dir, train_config.checkpoint_retention_policy.max_to_keep)

        if eval_period > 0 and (iteration + 1) % eval_period == 0 and iteration != max_iter - 1:
            val_results_dict = val_evaluator.evaluate_and_maybe_save(
                feature_model=feature_model,
                linear_classifiers=linear_classifiers,
                prefixstring=f"ITER: {iteration}",
                iteration=iteration,
            )
            val_accuracy = val_results_dict[val_evaluator.main_metric_name]
            if val_accuracy >= best_accuracy:
                best_accuracy = val_accuracy
                (ckpt_dir / "best").mkdir(parents=True, exist_ok=True)
                checkpoint = {
                    "iteration": iteration,
                    "linear_classifiers": linear_classifiers.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_accuracy": best_accuracy,
                }
                torch.save(checkpoint, ckpt_dir / "best" / "checkpoint.pth")
            torch.distributed.barrier()

        iteration = iteration + 1

    return feature_model, linear_classifiers, iteration


def make_train_transform(config):
    train_transform = make_classification_train_transform(crop_size=config.crop_size)
    return train_transform


def make_train_dataset(train_dataset: str, transform_config):
    train_transform = make_train_transform(transform_config)
    return make_dataset(dataset_str=train_dataset, transform=train_transform)


def eval_linear_with_model(*, model: torch.nn.Module, autocast_dtype, config):
    start = time.time()
    cudnn.benchmark = True

    train_dataset = make_train_dataset(config.train.dataset, config.transform)
    training_num_classes = get_num_classes(train_dataset)
    train_dataset_dict = create_train_dataset_dict(
        train_dataset,
        few_shot_eval=config.few_shot.enable,
        few_shot_k_or_percent=config.few_shot.k_or_percent,
        few_shot_n_tries=config.few_shot.n_tries,
    )
    n_last_blocks = config.train.n_last_blocks
    autocast_ctx = partial(torch.autocast, device_type="cuda", enabled=True, dtype=autocast_dtype)
    feature_model = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx)

    save_results_func = None
    if config.save_results:
        save_results_func = partial(default_save_results_func, output_dir=config.output_dir)

    metrics_file_path = os.path.join(config.output_dir, config.metrics_file_name)
    val_evaluator, test_evaluators = make_evaluators(
        eval_config=config.eval,
        val_metric_type=config.train.val_metric_type,
        val_dataset=config.train.val_dataset,
        transform_config=config.transform,
        metrics_file_path=metrics_file_path,
        training_num_classes=training_num_classes,
        save_results_func=save_results_func,
    )
    results_dict = {}
    checkpoint_output_dirs: list = []
    for _try in train_dataset_dict.keys():
        if len(train_dataset_dict) > 1:
            checkpoint_output_dir = os.path.join(config.output_dir, f"checkpoints_{_try}")
            save_filename_suffix = f"_{_try}"
        else:
            checkpoint_output_dir = os.path.join(config.output_dir, "checkpoints")
            save_filename_suffix = ""
        os.makedirs(checkpoint_output_dir, exist_ok=True)

        feature_model, linear_classifiers, iteration = train_linear_classifiers(
            feature_model=feature_model,
            train_dataset=train_dataset_dict[_try],
            train_config=config.train,
            training_num_classes=training_num_classes,
            val_evaluator=val_evaluator,
            checkpoint_output_dir=checkpoint_output_dir,
        )
        checkpoint_output_dirs.append(checkpoint_output_dir)
        results_dict[_try] = val_evaluator.evaluate_and_maybe_save(
            feature_model=feature_model,
            linear_classifiers=linear_classifiers,
            iteration=iteration,
            save_filename_suffix=save_filename_suffix,
        )
        for test_evaluator in test_evaluators:
            eval_results_dict = test_evaluator.evaluate_and_maybe_save(
                feature_model=feature_model,
                linear_classifiers=linear_classifiers,
                iteration=iteration,
                best_classifier_on_val=results_dict[_try]["best_classifier"],
                save_filename_suffix=save_filename_suffix,
            )
            results_dict[_try] = {**eval_results_dict, **results_dict[_try]}

    if len(train_dataset_dict) > 1:
        results_dict = average_metrics(results_dict, ignore_keys=["best_classifier"])
    else:
        results_dict = {**results_dict[_try]}

    for checkpoint_output_dir in checkpoint_output_dirs:
        if distributed.is_subgroup_main_process():
            cleanup_checkpoint(checkpoint_output_dir, config.train.checkpoint_retention_policy)

    logger.info("Test Results Dict " + str(results_dict))
    logger.info(f"Linear evaluation done in {int(time.time() - start)}s")
    return results_dict


def main(argv=None):
    args = _parse_args(argv)
    config = _build_runtime_config(args)
    os.makedirs(config.output_dir, exist_ok=True)
    model, model_context = load_model_and_context(config.model, output_dir=config.output_dir)
    results_dict = eval_linear_with_model(
        model=model, autocast_dtype=model_context["autocast_dtype"], config=config
    )
    _write_results_file(results_dict, config.output_dir)
    logger.info("Evaluation complete: %s", results_dict)
    return results_dict


if __name__ == "__main__":
    main()

"""
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=1 /project/dinov3/dino_transfer_classification.py \
  --config-file /project/dinov3/dinov3/configs/ssl_default_config.yaml \
  --pretrained-weights /project/dinov3/ckpt/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --train-dataset "/project/data/external/ILSVRC/Data/CLS-LOC/val" \
  --val-dataset "/project/data/external/ILSVRC/Data/CLS-LOC/val" \
  --output-dir /project/results/dinov3/linear_eval \
  --learning-rate 0.001 \
  --n-last-blocks 1 \
  --epochs 10 \
  --epoch-length 1250
"""