"""Classes for implementing data augmentation pipelines.

Authors
 * Mirco Ravanelli 2022
"""

import logging
import random

import torch
import torch.nn.functional as F

from speechbrain.utils.callchains import lengths_arg_exists
from DQN_agent import *
logger = logging.getLogger(__name__)

from DQN_agent import *
class rl_Augmenter(torch.nn.Module):
    def __init__(
        self,
        num_augmentations=1,
    ):
        # super().__init__()
        super(torch.nn.Module, self).__init__()
        self.parallel_augment = False
        self.parallel_augment_fixed_bs = False
        self.concat_original = True
        self.augmentations = list()
        self.min_augmentations = 1
        self.max_augmentations = 10
        self.shuffle_augmentations =True
        self.augment_start_index = 0
        self.augment_end_index = None
        self.concat_start_index = 0
        self.concat_end_index = None
        self.repeat_augment = 1
        self.N_augment=num_augmentations

        # Check min and max augmentations
        self.check_min_max_augmentations()

        # This variable represents the total number of augmentations to perform for each signal,
        # including the original signal in the count.

        #------agent----------------
        self.num_augmentations =num_augmentations
        self.do_augment = True

        
        # Check repeat augment arguments
        if not isinstance(self.repeat_augment, int):
            raise ValueError("repeat_augment must be an integer.")

        if self.repeat_augment < 0:
            raise ValueError("repeat_augment must be greater than 0.")

        if self.augment_end_index is not None:
            if self.augment_end_index < self.augment_start_index:
                raise ValueError(
                    "augment_end_index must be smaller or equal to augment_start_index."
                )

        if self.concat_end_index is not None:
            if self.concat_end_index < self.concat_start_index:
                raise ValueError(
                    "concat_end_index must be smaller or equal to concat_start_index."
                )
        # Turn augmentations into a dictionary
        self.augmentations={}


        if len(self.augmentations) == 0:
            logger.warning(
                "No augmentation is applied because the augmentation list is empty."
            )

        # Check min and max augmentations
        if self.max_augmentations <= 0:
            logger.warning(
                "No augmentations applied because max_augmentations is non-positive."
            )
        if self.min_augmentations < 0:
            self.min_augmentations = 0
            logger.warning(
                "min_augmentations is negative. Modified to be non-negative."
            )
        if self.min_augmentations > self.max_augmentations:
            logger.warning(
                "min_augmentations is greater than max_augmentations. min_augmentations set to max_augmentations."
            )
            self.max_augmentations = self.min_augmentations

        # Check if augmentation modules need the length argument
        self.require_lengths = {}

    def augment(self, x, lengths, selected_augmentations):
        next_input = x
        next_lengths = lengths
        output = []
        output_lengths = []
        out_lengths = lengths
        for k, augment_name in enumerate(selected_augmentations):
            augment_fun = self.augmentations[augment_name]

            idx = torch.arange(x.shape[0])
            if self.parallel_augment and self.parallel_augment_fixed_bs:
                idx_startstop = torch.linspace(
                    0, x.shape[0], len(selected_augmentations) + 1
                ).to(torch.int)
                idx_start = idx_startstop[k]
                idx_stop = idx_startstop[k + 1]
                idx = idx[idx_start:idx_stop]

            # Check input arguments
            if self.require_lengths[augment_name]:
                out = augment_fun(
                    next_input[idx, ...], lengths=next_lengths[idx]
                )
            else:
                out = augment_fun(next_input[idx, ...])

            # Check output arguments
            if isinstance(out, tuple):
                if len(out) == 2:
                    out, out_lengths = out
                else:
                    raise ValueError(
                        "The function must return max two arguments (Tensor, Length[optional])"
                    )

            # Manage sequential or parallel augmentation
            if not self.parallel_augment:
                next_input = out
                next_lengths = out_lengths[idx]
            else:
                output.append(out)
                output_lengths.append(out_lengths)

        if self.parallel_augment:
            # Concatenate all the augmented data
            output, output_lengths = self.concatenate_outputs(
                output, output_lengths
            )
        else:
            # Take the last augmented signal of the pipeline
            output = out
            output_lengths = out_lengths

        return output, output_lengths

    def forward(self, x, lengths,selected_aug):

        x_original = x
        len_original = lengths
        # Determine the ending index for augmentation, considering user-specified or default values.
        self.augment_end_index_batch = (x.shape[0])

        self.augmentations = {augmentation.__class__.__name__ + str(i): augmentation for i, augmentation in enumerate(selected_augmentations)}
        for aug_key, aug_fun in self.augmentations.items():
            self.require_lengths[aug_key] = lengths_arg_exists(aug_fun.forward)


        if self.augment_start_index >= x.shape[0]:
            self.do_augment = False
            logger.warning(
                "No augmentation is applied because the augmentation start index is greater than or equal to the number of examples in the input batch."
            )
            return x, lengths
                # No augmentation
        if (
            self.repeat_augment == 0
            or self.N_augment == 0
            or len(augmentations_lst) == 0
        ):
            self.do_augment = False
            return x, lengths
        

        # Get augmentations list
        augmentations_lst = selected_aug

        if self.shuffle_augmentations:
            random.shuffle(augmentations_lst)
        selected_augmentations = augmentations_lst[0 : self.N_augment]

        # Select the portion of the input to augment and update lengths accordingly.
        x = x[self.augment_start_index : self.augment_end_index_batch]
        lengths = lengths[self.augment_start_index : self.augment_end_index_batch]

        # Lists to collect the outputs
        output_lst = []
        output_len_lst = []

        # Concatenate the original signal if required
        self.skip_concat = not (self.concat_original)
        if self.concat_original:
            # Check start index
            if self.concat_start_index >= x.shape[0]:
                self.skip_concat = True
                pass
            else:
                self.skip_concat = False
                # Determine the ending index for concatenation, considering user-specified or default values.
                self.concat_end_index_batch = (
                    min(self.concat_end_index, x_original.shape[0])
                    if self.concat_end_index is not None
                    else x_original.shape[0]
                )

                output_lst.append(
                    x_original[
                        self.concat_start_index : self.concat_end_index_batch
                    ]
                )
                output_len_lst.append(
                    len_original[
                        self.concat_start_index : self.concat_end_index_batch
                    ]
                )

        # Perform augmentations
        for i in range(self.repeat_augment):
            output, output_lengths = self.augment(
                x, lengths, selected_augmentations
            )
            output_lst.append(output)
            output_len_lst.append(output_lengths)

        # Concatenate the final outputs while handling scenarios where
        # different temporal dimensions may arise due to augmentations
        # like speed change.
        output, output_lengths = self.concatenate_outputs(
            output_lst, output_len_lst
        )




        return output, output_lengths

    def concatenate_outputs(self, augment_lst, augment_len_lst):

        # Find the maximum temporal dimension (batch length) among the sequences
        max_len = max(augment.shape[1] for augment in augment_lst)

        # Rescale the sequence lengths to adjust for augmented batches with different temporal dimensions.
        augment_len_lst = [
            length * (output.shape[1] / max_len)
            for length, output in zip(augment_len_lst, augment_lst)
        ]

        # Pad sequences to match the maximum temporal dimension.
        # Note that some augmented batches, like those with speed changes, may have different temporal dimensions.
        augment_lst = [
            F.pad(output, (0, max_len - output.shape[1]))
            for output in augment_lst
        ]

        # Concatenate the padded sequences and rescaled lengths
        output = torch.cat(augment_lst, dim=0)
        output_lengths = torch.cat(augment_len_lst, dim=0)

        return output, output_lengths

    def replicate_multiple_labels(self, *args):
    
        # Determine whether to apply data augmentation
        if not self.do_augment:
            return args

        list_of_augmented_labels = []

        for labels in args:
            list_of_augmented_labels.append(self.replicate_labels(labels))

        return list_of_augmented_labels

    def replicate_labels(self, labels):

        # Determine whether to apply data augmentation
        if not self.do_augment:
            return labels

        augmented_labels = []
        if self.concat_original and not (self.skip_concat):
            augmented_labels = [
                labels[self.concat_start_index : self.concat_end_index_batch]
            ]
        selected_labels = labels[
            self.augment_start_index : self.augment_end_index_batch
        ]

        if self.parallel_augment:
            selected_labels = torch.cat(
                [selected_labels] * self.N_augment, dim=0
            )

        augmented_labels = (
            augmented_labels + [selected_labels] * self.repeat_augment
        )

        augmented_labels = torch.cat(augmented_labels, dim=0)

        return augmented_labels

    def check_min_max_augmentations(self):
        """Checks the min_augmentations and max_augmentations arguments."""
        if self.min_augmentations is None:
            self.min_augmentations = 1
        if self.max_augmentations is None:
            self.max_augmentations = len(self.augmentations)
        if self.max_augmentations > len(self.augmentations):
            self.max_augmentations = len(self.augmentations)
        if self.min_augmentations > len(self.augmentations):
            self.min_augmentations = len(self.augmentations)
