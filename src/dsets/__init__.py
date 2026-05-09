from typing import Callable

from dsets.imagenet import get_imagenet, get_cub, get_food

DATASETS = {
    "imagenet": get_imagenet,
    "cub": get_cub,
    "food": get_food,
}

def get_dataset(dataset_name: str) -> Callable:
    """
    Get dataset by name.
    :param dataset_name: Name of the dataset.
    :return: Dataset.
    """
    if dataset_name in DATASETS:
        dataset = DATASETS[dataset_name]
        print(f"Loading {dataset_name}")
        return dataset
    else:
        raise KeyError(f"DATASET {dataset_name} not defined.")