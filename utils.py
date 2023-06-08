"""
Author: c0ldstudy
2022-03-29 13:44:53
"""

import os
import torch
from torchvision import datasets, transforms
import numpy as np
import collections
import random

class TorchVisionDataset:
    """
    - name: the dataset name
    - subset: the subset of the main dataset. Dataset will be loaded as ``nlp.load_dataset(name, subset)``.
    - label_map: Mapping if output labels should be re-mapped. Useful
      if model was trained with a different label arrangement than
      provided in the ``nlp`` version of the dataset.
    - output_scale_factor (float): Factor to divide ground-truth outputs by.
        Generally, TextAttack goal functions require model outputs
        between 0 and 1. Some datasets test the model's correlation
        with ground-truth output, instead of its accuracy, so these
        outputs may be scaled arbitrarily.
    - shuffle (bool): Whether to shuffle the dataset on load.
    """

    def __init__(
        self, name,data, split="train", shuffle=True,
    ):
        self._name = name
        self._split = split
        self._dataset = data

        # Input/output column order, like (('premise', 'hypothesis'), 'label')

        self.input_columns, self.output_column = ("image", "label")

        self._i = 0
        self.examples = list(self._dataset)

        if shuffle:
            random.shuffle(self.examples)

    def __len__(self):
        return len(self._dataset)

    def _format_raw_example(self, raw_example):
        return raw_example

    def __next__(self):
        if self._i >= len(self.examples):
            raise StopIteration
        raw_example = self.examples[self._i]
        self._i += 1
        return self._format_raw_example(raw_example)

    def __getitem__(self, i):
        if isinstance(i, int):
            return self._format_raw_example(self.examples[i])
        else:
            # `i` could be a slice or an integer. if it's a slice,
            # return the formatted version of the proper slice of the list
            return [self._format_raw_example(ex) for ex in self.examples[i]]

    def get_json_data(self):
        if self.examples:
            new_data = []
            for idx, instance in enumerate(self.examples):
                new_instance = {}
                new_instance["image"] = instance[0].numpy().tolist()
                new_instance["label"] = instance[1]
                new_instance["uid"] = idx
                new_data.append(new_instance)
        return new_data

def get_dataset(dataset_configs):
    if dataset_configs['name'] == "MNIST":
        return _read_mnist_dataset(dataset_configs, "MNIST")
    elif dataset_configs['name'] == "CIFAR10":
        return _read_cifar10_dataset(dataset_configs, "CIFAR10")

def _split_by_labels(num_classes, train_data, server_number_sampled, train_server_path):
    subset_indices = []
    for i in range(num_classes):
        indices_xi = (torch.LongTensor(train_data.targets) == i).nonzero(as_tuple=True)[0]
        sampled_indices = np.random.choice(
            indices_xi, server_number_sampled, replace=False
        )
        subset_indices.extend(sampled_indices)
    train_server_subset = torch.utils.data.Subset(train_data, subset_indices)
    torch.save(train_server_subset, train_server_path)
    return train_server_subset

def label_update(num_classes, train_data, server_number_sampled, train_server_path, custom=False, label_custom=None):
    subset_indices = []

    if custom == True:
        for i in list(label_custom.keys()):
            for ori_label in label_custom[i]:
                single_number_sampled = int(server_number_sampled / len(label_custom[i]))
                # print(single_number_sampled)
                    # break
                indices_xi = (torch.LongTensor(train_data.targets) == ori_label).nonzero(as_tuple=True)[0]
                sampled_indices = np.random.choice(indices_xi, single_number_sampled, replace=False)
                subset_indices.extend(sampled_indices)

        train_data.targets = torch.tensor(train_data.targets)
        for i in list(label_custom.keys()):
            mask = sum(train_data.targets==i for i in label_custom[i]).bool()
            train_data.targets[mask] = i

        # mask = sum(train_data.targets==i for i in label_binary[0]).bool()
        # train_data.targets[mask] = 0
        # mask = sum(train_data.targets==i for i in label_binary[1]).bool()
        # train_data.targets[mask] = 1
        # mask = sum(train_data.targets==i for i in label_binary[2]).bool()
        # train_data.targets[mask] = 2
        # mask = sum(train_data.targets==i for i in label_binary[3]).bool()
        # train_data.targets[mask] = 3
    else:
        for i in range(num_classes):
            indices_xi = (torch.LongTensor(train_data.targets) == i).nonzero(as_tuple=True)[0]
            sampled_indices = np.random.choice(indices_xi, server_number_sampled, replace=False)
            subset_indices.extend(sampled_indices)


    train_server_subset = torch.utils.data.Subset(train_data, subset_indices)
    # print(len(train_server_subset))
    # print(np.unique(train_server_subset.targets, return_counts=True))
    # exit()
    torch.save(train_server_subset, train_server_path)
    return train_server_subset

def _read_mnist_dataset(dataset_configs, dataset_name):
    path = dataset_configs["dataset_path"]

    # 1.1 Training Data
    train_student_path = os.path.join(path, "train_split.pt")
    if os.path.exists(train_student_path):
        train_student_subset = torch.load(train_student_path)
    else:
        train_data = datasets.MNIST(
            root=path,
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(),]),
        )
        num_classes = len(train_data.classes)
        student_number_sampled = dataset_configs["student_train_number"] // num_classes
        train_student_subset = _split_by_labels(
            num_classes, train_data, student_number_sampled, train_student_path
        )
    train_student_data = TorchVisionDataset(
        name=dataset_name, data=train_student_subset, split="train",
    )

    # 1.2 Test Data
    test_student_path = os.path.join(path, "test_split.pt")
    if os.path.exists(test_student_path):
        test_student_subset = torch.load(test_student_path)
    else:
        test_data = datasets.MNIST(
            root=path,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(),]),
        )
        num_classes = len(test_data.classes)
        student_number_sampled = dataset_configs["student_test_number"] // num_classes
        test_student_subset = _split_by_labels(
            num_classes, test_data, student_number_sampled, test_student_path
        )
    test_student_data = TorchVisionDataset(
        name=dataset_name, data=test_student_subset, split="test",
    )

    # 1.3 Val Data
    val_student_path = os.path.join(path, "val_split.pt")
    if os.path.exists(val_student_path):
        val_student_subset = torch.load(val_student_path)
    else:
        val_data = datasets.MNIST(
            root=path,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(),]),
        )
        num_classes = len(val_data.classes)
        student_number_sampled = dataset_configs["student_val_number"] // num_classes
        val_student_subset = _split_by_labels(
            num_classes, val_data, student_number_sampled, val_student_path
        )
    val_student_data = TorchVisionDataset(
        name=dataset_name, data=val_student_subset, split="test",
    )

    print(f"train_data length: {len(train_student_data)}, test_data length: {len(test_student_data)}, val_data length: {len(val_student_data)}")
    return {
        "train": train_student_data,
        "test": test_student_data,
        "val": val_student_data,
    }


def _read_cifar10_dataset(dataset_configs, dataset_name):
    path = dataset_configs["dataset_path"]
    custom = dataset_configs["binary"]

    # label_to_str = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'} Just for information
    label_custom = {0: [0,8], 1: [1,9], 2:[2,6,4], 3:[3,5,7]}
    label_student = {2:[2,6,4], 3:[3,5,7]}
    label_server= {2:[2,6,4], 3:[3,5,7]}

    # 1.2 Training Data for Student
    train_student_path = os.path.join(path, "train_split.pt")

    if os.path.exists(train_student_path):
        train_student_subset = torch.load(train_student_path)
    else:
        train_data = datasets.CIFAR10(
            root=path,
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )

        # num_classes = len(train_data.classes)
        # student_number_sampled = dataset_configs["student_train_number"] // num_classes
        # train_student_subset = _split_by_labels(
        #     num_classes, train_data, student_number_sampled, train_student_path
        # )
        num_classes = len(label_custom)
        student_number_sampled = dataset_configs["student_train_number"] // num_classes
        train_student_subset = label_update(
            num_classes, train_data, student_number_sampled, train_student_path, custom, label_custom)

    # print(len(train_student_subset))
    #     # print(train_data.max)
    #     # print(train_data.min)
    # print(torch.max(((train_student_subset[0][0]))))
    # print(torch.min(((train_student_subset[0][0]))))
    # exit()
    train_student_data = TorchVisionDataset(
        name=dataset_name, data=train_student_subset, split="train",
    )

    # 1.3 Validation Data for Student
    val_student_path = os.path.join(path, "val_split.pt")
    if os.path.exists(val_student_path):
        val_student_subset = torch.load(val_student_path)
    else:
        val_data = datasets.CIFAR10(
            root=path,
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )
        # num_classes = len(val_data.classes)
        # student_number_sampled = dataset_configs["student_val_number"] // num_classes
        # val_student_subset = _split_by_labels(
        #     num_classes, val_data, student_number_sampled, val_student_path
        # )
        num_classes = len(label_custom)
        student_number_sampled = dataset_configs["student_val_number"] // num_classes
        val_student_subset = label_update(
            num_classes, val_data, student_number_sampled, val_student_path, custom, label_custom)

    val_student_data = TorchVisionDataset(
        name=dataset_name, data=val_student_subset, split="val",
    )



    # 2.2 Test Data for Student
    test_student_path = os.path.join(path, "test_split.pt")
    if os.path.exists(test_student_path):
        test_student_subset = torch.load(test_student_path)
    else:
        test_data = datasets.CIFAR10(
            root=path,
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )
        # num_classes = len(test_data.classes)
        # student_number_sampled = dataset_configs["student_test_number"] // num_classes
        # test_student_subset = _split_by_labels(
        #     num_classes, test_data, student_number_sampled, test_student_path
        # )
        num_classes = len(label_student)
        student_number_sampled = dataset_configs["student_test_number"] // num_classes
        test_student_subset = label_update(
            num_classes, test_data, student_number_sampled, test_student_path, custom, label_student)



    test_student_data = TorchVisionDataset(
        name=dataset_name, data=test_student_subset, split="test",
    )

    print(
        f"train_student_data length: {len(train_student_data)}, val_student_data length: {len(val_student_data)}, test_student_data length: {len(test_student_data)}"
    )
    return {
        "train": train_student_data,
        "val": val_student_data,
        "test": test_student_data,
    }
