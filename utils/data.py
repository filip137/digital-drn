from __future__ import annotations

from torchvision import transforms


_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def build_image_transforms(dataset_name: str, *, train: bool) -> transforms.Compose:
    if dataset_name == "mnist":
        return transforms.Compose([transforms.ToTensor()])

    if dataset_name == "cifar10":
        ops: list[object] = []
        if train:
            ops.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4, padding_mode="edge"),
                ]
            )
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
            ]
        )
        return transforms.Compose(ops)

    raise ValueError(f"Unsupported dataset '{dataset_name}'.")
