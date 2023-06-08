from typing import List

import numpy as np
import torch
from copy import deepcopy

np.random.seed(1901)

class Attack:
    def __init__(
        self,
        vm, device, attack_path,
        epsilon = 0.214,
        min_val = 0,
        max_val = 1
    ):
        """
        This file contains code for untargeted FGSM attack
        args:
            vm: virtual model is wrapper used to get outputs/gradients of a model.
            device: system on which code is running "cpu"/"cuda"
            epsilon: magnitude of perturbation that is added

        """
        self.vm = vm
        self.device = device
        self.attack_path = attack_path
        self.epsilon = epsilon
        self.min_val = 0
        self.max_val = 1

    def attack(
        self, original_images: np.ndarray, labels: List[int], target_label = None,
    ):
        original_images = original_images.to(self.device)
        # original_images = torch.unsqueeze(original_images, 0)
        labels = torch.tensor(labels).to(self.device)
        # target_labels = target_label * torch.ones_like(labels).to(self.device)
        # print(original_images.shape)
        # get gradient with repect to labels
        original_images.requires_grad = True
        # print(labels)
        data_grad = self.vm.get_batch_input_gradient(original_images, labels)
        sign_data_grad = data_grad.sign()

        # perturd image
        perturbed_image = original_images + self.epsilon*sign_data_grad

        perturbed_image = torch.clamp(perturbed_image, self.min_val, self.max_val)
        adv_outputs, _ = self.vm.get_batch_output(perturbed_image)
        final_pred = adv_outputs.max(1, keepdim=True)[1]
        correct = 0
        correct += (final_pred != labels).sum().item()
        return perturbed_image.cpu().detach().numpy(), correct
