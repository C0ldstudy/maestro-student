from typing import List, Iterator, Dict, Tuple, Any, Type

import numpy as np
import torch
from copy import deepcopy

np.random.seed(1901)

class Attack:
    def __init__(
        self,
        vm, device, attack_path,
        min_val = 0,
        max_val = 1
    ):
        """
        args:
            vm: virtual model is wrapper used to get outputs/gradients of a model.
            device: system on which code is running "cpu"/"cuda"
            min_val: minimum value of each element in original image
            max_val: maximum value of each element in original image
                     each element in perturbed image should be in the range of min_val and max_val
            attack_path: Any other sources files that you may want to use like models should be available in ./submissions/ folder and loaded by attack.py.
                         Server doesn't load any external files. Do submit those files along with attack.py
        """
        self.vm = vm
        self.device = device
        self.attack_path = attack_path
        self.min_val = 0
        self.max_val = 1

    def attack(
        self, original_images: np.ndarray, labels: List[int], target_label = None,
    ):
        original_images = original_images.to(self.device)
        # original_images = torch.unsqueeze(original_images, 0)
        labels = torch.tensor(labels).to(self.device)
        target_labels = target_label * torch.ones_like(labels).to(self.device)
        perturbed_image = original_images

        # -------------------- TODO ------------------ #

        # Write your attack function here

        # ------------------ END TODO ---------------- #

        adv_outputs, _ = self.vm.get_batch_output(perturbed_image)
        final_pred = adv_outputs.max(1, keepdim=True)[1]
        correct = 0
        correct += (final_pred == target_labels).sum().item()
        return perturbed_image.cpu().detach().numpy(), correct
