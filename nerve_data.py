# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Small library that points to the nerve segmentation data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from NerveSegmentation.dataset import Dataset


class NerveData(Dataset):
    """Nerve Segmentation data set."""

    def __init__(self, subset):
        super(NerveData, self).__init__('Nerve', subset)

    def num_classes(self):
        """Returns the number of classes in the data set."""
        return 2

    def num_examples_per_epoch(self):
        """
        Returns the number of examples in the data subset.
        The number depends on the train_split_percent chosen
        in the build_dataset.py script.
        """
        if self.subset == 'train':
            return 4508
        if self.subset == 'test':
            return 1127

    def dataset_creation_message(self):
        """Instruction to build dataset."""

        print('Failed to find any Nerve Segmentation image %s files' % self.subset)
        print('')
        print('If you have already downloaded and processed the data, then make '
              'sure to set --data_dir to point to the directory containing the '
              'location of the sharded TFRecords.\n')
        print('Please see README.md for instructions on how to build '
              'the nerve segmentation dataset using build_dataset.py.\n')
