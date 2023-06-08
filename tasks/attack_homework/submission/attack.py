from typing import List, Iterator, Dict, Tuple, Any, Type

import numpy as np
import torch
from copy import deepcopy

np.random.seed(1901)

class Attack:
    def __init__(
        self,
        vm, device, attack_path,
        image_size = [1,28,28],
        n_population=100,
        n_generation=100,
        mask_rate=0.2,
        temperature=0.3,
        use_mask=True,
        step_size=0.1,
        child_rate=0.5,
        mutate_rate=0.6,
        l2_threshold=7.5,
    ):
        """
        args:
            vm: virtual model is wrapper used to get outputs/gradients of a model.
            image_size: [1,28,28]
            n_population: number of population in each iteration
            n_generation: maximum of generation constrained. The attack automatically stops when this maximum is reached
            mutate_rate: if use_mask is set to true, this is used to set the rate of masking when perturbed
            temperature: this sets the temperature when computing the probabilities for next generation
            use_mask: when this is true, only a subset of the image will be perturbed.
            l2_threshold: the constrain on the distance between the original data point and adversarial data point.
        """
        self.vm = vm
        self.device = device
        self.attack_path = attack_path
        self.image_size = image_size
        self.n_population = n_population
        self.n_generation = n_generation
        self.mask_rate = mask_rate
        self.temperature = temperature
        self.use_mask = use_mask
        self.step_size = step_size
        self.child_rate = child_rate
        self.mutate_rate = mutate_rate
        self.l2_threshold = l2_threshold

    def attack(
        self, original_image: np.ndarray, labels: List[int], target_label = None,
    ):
        """
        currently this attack has 2 versions, 1 with no mask pre-defined, 1 with mask pre-defined.
        args:
            original_image: a numpy ndarray images, [1,28,28]
            labels: label of the image, a list of size 1
            target_label: target label we want the image to be classified, int
        return:
            the perturbed image
            label of that perturbed iamge
            success: whether the attack succeds
        """
        original_image = torch.squeeze(original_image, 0)
        original_image = original_image.cpu().detach().numpy()
        self.image_size = original_image.shape
        self.original_image = np.array(original_image)
        # print("image_size = ", self.original_image.shape)
        # print("target_label is: ",target_label)
        self.mask = np.random.binomial(1, self.mask_rate, size=self.image_size).astype(
            "bool"
        )
        population = self.init_population(original_image)
        # print("type of population: ", type(population))
        examples = [(labels[0], labels[0], np.squeeze(x)) for x in population[:10]]
        #visualize(examples, "population.png")
        success = False
        for g in range(self.n_generation):
            # print("generation: ", g)
            population, output, scores, best_index = self.eval_population(
                population, target_label
            )
            # type(population) : <class 'numpy.ndarray'>
            # print("shape of population: ", population)
            # print("shape of population: ", population.shape)
            # print(f"Generation: {g} best score: {scores[best_index]}")
            if np.argmax(output[best_index, :]) == target_label:
                print(f"Attack Success!")
                # visualize([(labels[0],np.argmax(output[best_index, :]),np.squeeze(population[best_index]))], "after_GA1.png")
                success = True
                break
        # print("type of population: ", type(population[best_index]))

        perturbed_image = population[best_index]

        # print("shape of population 1: ", perturbed_image.shape)
        perturbed_image = np.expand_dims(perturbed_image,0)
        return perturbed_image, success

    def fitness(self, image: np.ndarray, target: int):
        """
        evaluate how fit the current image is
        return:
            output: output of the model
            scores: the "fitness" of the image, measured as logits of the target label
        """
        output = self._get_batch_outputs_numpy(image)
        softmax_output = np.exp(output) / np.expand_dims(
            np.sum(np.exp(output), axis=1), axis=1
        )
        scores = softmax_output[:, target]
        return output, scores

    def eval_population(self, population, target_label):
        """
        evaluate the population, pick the parents, and then crossover to get the next
        population
        args:
            population: current population, a list of images
            target_label: target label we want the imageto be classiied, int
        return:
            population: population of all the images
            output: output of the model
            scores: the "fitness" of the image, measured as logits of the target label
            best_indx: index of the best image in the population
        """
        output, scores = self.fitness(population, target_label)
        # --------------TODO--------------
        score_ranks = None  # Sort the scores from largeset to smallest
        best_index = 0  # The index for the best scored candidate
        logits = None  # Exponentiate the scores after incorporating temperature
        select_probs = None  # Normalize the logits between 0-1
        # ------------END TODO-------------

        if np.argmax(output[best_index, :]) == target_label:
            return population, output, scores, best_index

        # --------------TODO--------------
        # Compute the next generation of population, which is comprised of Elite, Survived, and Offspirngs
        # Elite: top scoring gene, will not be mutated
        elite = []

        # Survived: rest of the top genes that survived, mutated with some probability
        survived = []  # Survived, and mutate some of them

        # Offsprings: offsprings of strong genes
        # Identify the parents of the children based on select_probs, then use crossover to produce the next generation
        children = []

        # population =np.array(elite + survived +children)
        population = population  # Delete this and uncomment the line above if you finished implementing
        # ------------END TODO-------------
        return population, output, scores, best_index

    def perturb(self, image):
        """
        perturb a single image with some constraints and a mask
        args:
            image: the image to be perturbed
        return:
            perturbed: perturbed image
        """
        if not self.use_mask:
            adv_images = image + np.random.randn(*self.mask.shape) * self.step_size
            # perturbed = np.maximum(np.minimum(adv_images,self.original_image+0.5), self.original_image-0.5)
            delta = np.expand_dims(adv_images - self.original_image, axis=0)
            # Assume x and adv_images are batched tensors where the first dimension is
            # a batch dimension
            eps = self.l2_threshold
            mask = (
                np.linalg.norm(delta.reshape((delta.shape[0], -1)), ord=2, axis=1)
                <= eps
            )
            scaling_factor = np.linalg.norm(
                delta.reshape((delta.shape[0], -1)), ord=2, axis=1
            )
            scaling_factor[mask] = eps
            delta *= eps / scaling_factor.reshape((-1, 1, 1, 1))
            perturbed = self.original_image + delta
            perturbed = np.squeeze(np.clip(perturbed, 0, 1), axis=0)
        else:
            perturbed = np.clip(
                image + self.mask * np.random.randn(*self.mask.shape) * self.step_size,
                0,
                1,
            )
        # print("size of perturbed :", perturbed.shape)
        return perturbed

    def crossover(self, x1, x2):
        """
        crossover two images to get a new one. We use a uniform distribution with p=0.5
        args:
            x1: image #1
            x2: image #2
        return:
            x_new: newly crossovered image
        """
        x_new = x1.copy()
        for i in range(x1.shape[1]):
            for j in range(x1.shape[2]):
                if np.random.uniform() < 0.5:
                    x_new[0][i][j] = x2[0][i][j]
        return x_new

    def init_population(self, original_image: np.ndarray):
        """
        Initialize the population to n_population of images. Make sure to perturbe each image.
        args:
            original_image: image to be attacked
        return:
            a list of perturbed images initialized from orignal_image
        """
        return np.array(
            [self.perturb(original_image[0]) for _ in range(self.n_population)]
        )

    def _get_batch_outputs_numpy(self, image: np.ndarray):
        image_tensor = torch.FloatTensor(image)
        image_tensor = image_tensor.to(self.device)

        outputs, _ = self.vm.get_batch_output(image_tensor)

        return outputs.cpu().detach().numpy()
