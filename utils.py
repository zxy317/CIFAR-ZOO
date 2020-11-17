# -*-coding:utf-8-*-
import logging
import math
import os
import shutil
import tensorflow as tf
from scipy.stats import ttest_ind
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class Logger(object):
    def __init__(self, log_file_name, log_level, logger_name):
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)
        file_handler = logging.FileHandler(log_file_name)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] - [%(filename)s line:%(lineno)3d] : %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def data_augmentation(config, is_train=True):
    aug = []
    if is_train:
        # random crop
        if config.augmentation.random_crop:
            aug.append(transforms.RandomCrop(config.input_size, padding=4))
        # horizontal filp
        if config.augmentation.random_horizontal_filp:
            aug.append(transforms.RandomHorizontalFlip())

    aug.append(transforms.ToTensor())
    # normalize  [- mean / std]
    if config.augmentation.normalize:
        if config.dataset == "cifar10":
            aug.append(
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            )
        else:
            aug.append(
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            )

    if is_train and config.augmentation.cutout:
        # cutout
        aug.append(
            Cutout(n_holes=config.augmentation.holes, length=config.augmentation.length)
        )
    return aug


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename + ".pth.tar")
    if is_best:
        shutil.copyfile(filename + ".pth.tar", filename + "_best.pth.tar")


def load_checkpoint(path, model, optimizer=None):
    if os.path.isfile(path):
        logging.info("=== loading checkpoint '{}' ===".format(path))

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["state_dict"], strict=False)

        if optimizer is not None:
            best_prec = checkpoint["best_prec"]
            last_epoch = checkpoint["last_epoch"]
            optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(
                "=== done. also loaded optimizer from "
                + "checkpoint '{}' (epoch {}) ===".format(path, last_epoch + 1)
            )
            return best_prec, last_epoch


def get_data_loader(transform_train, transform_test, config):
    assert config.dataset == "cifar10" or config.dataset == "cifar100"
    if config.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root=config.data_path, train=True, download=True, transform=transform_train
        )

        testset = torchvision.datasets.CIFAR10(
            root=config.data_path, train=False, download=True, transform=transform_test
        )
    else:
        trainset = torchvision.datasets.CIFAR100(
            root=config.data_path, train=True, download=True, transform=transform_train
        )

        testset = torchvision.datasets.CIFAR100(
            root=config.data_path, train=False, download=True, transform=transform_test
        )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=config.test_batch, shuffle=False, num_workers=config.workers
    )
    return train_loader, test_loader


def mixup_data(x, y, alpha, device):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def adjust_learning_rate(optimizer, epoch, config):
    lr = get_current_lr(optimizer)
    if config.lr_scheduler.type == "STEP":
        if epoch in config.lr_scheduler.lr_epochs:
            lr *= config.lr_scheduler.lr_mults
    elif config.lr_scheduler.type == "COSINE":
        ratio = epoch / config.epochs
        lr = (
            config.lr_scheduler.min_lr
            + (config.lr_scheduler.base_lr - config.lr_scheduler.min_lr)
            * (1.0 + math.cos(math.pi * ratio))
            / 2.0
        )
    elif config.lr_scheduler.type == "HTD":
        ratio = epoch / config.epochs
        lr = (
            config.lr_scheduler.min_lr
            + (config.lr_scheduler.base_lr - config.lr_scheduler.min_lr)
            * (
                1.0
                - math.tanh(
                    config.lr_scheduler.lower_bound
                    + (
                        config.lr_scheduler.upper_bound
                        - config.lr_scheduler.lower_bound
                    )
                    * ratio
                )
            )
            / 2.0
        )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def flatten(nested_list):
    """Flatten a nested list."""
    return [item for a_list in nested_list for item in a_list]


def process_what_to_run_expand(pairs_to_test,
                               random_counterpart=None,
                               num_random_exp=100,
                               random_concepts=None):
    """Get concept vs. random or random vs. random pairs to run.

      Given set of target, list of concept pairs, expand them to include
       random pairs. For instance [(t1, [c1, c2])...] becomes
       [(t1, [c1, random1],
        (t1, [c1, random2],...
        (t1, [c2, random1],
        (t1, [c2, random2],...]

    Args:
      pairs_to_test: [(target, [concept1, concept2,...]),...]
      random_counterpart: random concept that will be compared to the concept.
      num_random_exp: number of random experiments to run against each concept.
      random_concepts: A list of names of random concepts for the random
                       experiments to draw from. Optional, if not provided, the
                       names will be random500_{i} for i in num_random_exp.

    Returns:
      all_concepts: unique set of targets/concepts
      new_pairs_to_test: expanded
    """

    def get_random_concept(i):
        return (random_concepts[i] if random_concepts
                else 'random500_{}'.format(i))

    new_pairs_to_test = []
    for (target, concept_set) in pairs_to_test:
        new_pairs_to_test_t = []
        # if only one element was given, this is to test with random.
        if len(concept_set) == 1:
            i = 0
            while len(new_pairs_to_test_t) < min(100, num_random_exp):
                # make sure that we are not comparing the same thing to each other.
                if concept_set[0] != get_random_concept(
                        i) and random_counterpart != get_random_concept(i):
                    new_pairs_to_test_t.append(
                        (target, [concept_set[0], get_random_concept(i)]))
                i += 1
        elif len(concept_set) > 1:
            new_pairs_to_test_t.append((target, concept_set))
        else:
            tf.logging.info('PAIR NOT PROCCESSED')
        new_pairs_to_test.extend(new_pairs_to_test_t)

    all_concepts = list(set(flatten([cs + [tc] for tc, cs in new_pairs_to_test])))

    return all_concepts, new_pairs_to_test


def process_what_to_run_concepts(pairs_to_test):
    """Process concepts and pairs to test.

    Args:
      pairs_to_test: a list of concepts to be tested and a target (e.g,
       [ ("target1",  ["concept1", "concept2", "concept3"]),...])

    Returns:
      return pairs to test:
         target1, concept1
         target1, concept2
         ...
         target2, concept1
         target2, concept2
         ...

    """

    pairs_for_sstesting = []
    # prepare pairs for concpet vs random.
    for pair in pairs_to_test:
        for concept in pair[1]:
            pairs_for_sstesting.append([pair[0], [concept]])
    return pairs_for_sstesting


def process_what_to_run_randoms(pairs_to_test, random_counterpart):
    """Process concepts and pairs to test.

    Args:
      pairs_to_test: a list of concepts to be tested and a target (e.g,
       [ ("target1",  ["concept1", "concept2", "concept3"]),...])
      random_counterpart: a random concept that will be compared to the concept.

    Returns:
      return pairs to test:
            target1, random_counterpart,
            target2, random_counterpart,
            ...
    """
    # prepare pairs for random vs random.
    pairs_for_sstesting_random = []
    targets = list(set([pair[0] for pair in pairs_to_test]))
    for target in targets:
        pairs_for_sstesting_random.append([target, [random_counterpart]])
    return pairs_for_sstesting_random


# helper functions to write summary files
def print_results(results, random_counterpart=None, random_concepts=None, num_random_exp=100,
                  min_p_val=0.05):
    """Helper function to organize results.
    If you ran TCAV with a random_counterpart, supply it here, otherwise supply random_concepts.
    If you get unexpected output, make sure you are using the correct keywords.

    Args:
      results: dictionary of results from TCAV runs.
      random_counterpart: name of the random_counterpart used, if it was used.
      random_concepts: list of random experiments that were run.
      num_random_exp: number of random experiments that were run.
      min_p_val: minimum p value for statistical significance
    """

    # helper function, returns if this is a random concept
    def is_random_concept(concept):
        if random_counterpart:
            return random_counterpart == concept

        elif random_concepts:
            return concept in random_concepts

        else:
            return 'random500_' in concept

    # print class, it will be the same for all
    print("Class =", results[0]['target_class'])

    # prepare data
    # dict with keys of concepts containing dict with bottlenecks
    result_summary = {}

    # random
    random_i_ups = {}

    for result in results:
        if result['cav_concept'] not in result_summary:
            result_summary[result['cav_concept']] = {}

        if result['bottleneck'] not in result_summary[result['cav_concept']]:
            result_summary[result['cav_concept']][result['bottleneck']] = []

        result_summary[result['cav_concept']][result['bottleneck']].append(result)

        # store random
        if is_random_concept(result['cav_concept']):
            if result['bottleneck'] not in random_i_ups:
                random_i_ups[result['bottleneck']] = []

            random_i_ups[result['bottleneck']].append(result['i_up'])

    # print concepts and classes with indentation
    for concept in result_summary:

        # if not random
        if not is_random_concept(concept):
            print(" ", "Concept =", concept)

            for bottleneck in result_summary[concept]:
                i_ups = [item['i_up'] for item in result_summary[concept][bottleneck]]

                # Calculate statistical significance
                _, p_val = ttest_ind(random_i_ups[bottleneck], i_ups)

                print(3 * " ", "Bottleneck =", ("%s. TCAV Score = %.2f (+- %.2f), "
                                                "random was %.2f (+- %.2f). p-val = %.3f (%s)") % (
                          bottleneck, np.mean(i_ups), np.std(i_ups),
                          np.mean(random_i_ups[bottleneck]),
                          np.std(random_i_ups[bottleneck]), p_val,
                          "not significant" if p_val > min_p_val else "significant"))


def make_dir_if_not_exists(directory):
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)
