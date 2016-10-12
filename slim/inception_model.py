# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model for Nerve Segmentation Challenge expressed in TensorFlow-Slim.

  Usage:

  # Parameters for BatchNorm.
  batch_norm_params = {
      # Decay for the batch_norm moving averages.
      'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
  }
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
    with slim.arg_scope([slim.ops.conv2d],
                        stddev=0.1,
                        activation=tf.nn.relu,
                        batch_norm_params=batch_norm_params):
      # Force all Variables to reside on the CPU.
      with slim.arg_scope([slim.variables.variable], device='/cpu:0'):
        logits, endpoints = slim.inception.inception_v3(
            images,
            dropout_keep_prob=0.8,
            num_classes=num_classes,
            is_training=for_training,
            restore_logits=restore_logits,
            scope=scope)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from NerveSegmentation.slim import ops
from NerveSegmentation.slim import scopes


def inception_v3(inputs,
                 dropout_keep_prob=0.8,
                 num_classes=2,
                 is_training=True,
                 restore_logits=True,
                 scope=''):
    """Latest Inception from http://arxiv.org/abs/1512.00567.

      "Rethinking the Inception Architecture for Computer Vision"

      Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens,
      Zbigniew Wojna

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      dropout_keep_prob: dropout keep_prob.
      num_classes: number of predicted classes.
      is_training: whether is training or not.
      restore_logits: whether or not the logits layers should be restored.
        Useful for fine-tuning a model with different num_classes.
      scope: Optional scope for op_scope.

    Returns:
      a list containing 'logits', 'aux_logits' Tensors.
    """
    # end_points will collect relevant activations for external use, for example
    # summaries or losses.
    end_points = {}
    with tf.op_scope([inputs], scope, 'nerve_net'):
        with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout],
                              is_training=is_training):
            with scopes.arg_scope([ops.conv2d, ops.max_pool], padding='SAME'):
                # 400 x 560 x 3
                end_points['conv0'] = ops.conv2d(inputs, 13, [3, 3], stride=2, scope='conv0')

                # 400 x 560 x 3
                end_points['pool1'] = ops.max_pool(inputs, [2, 2], stride=2, scope='pool1')

                # 200 x 280 x 16
                end_points['concat1'] = tf.concat(3, [end_points['conv0'], end_points['pool1']])

                net = ops.batch_norm(end_points['concat1'], prelu_activation=True, scope='batch_prelu1')
            # Nerve Net blocks
            with scopes.arg_scope([ops.conv2d, ops.max_pool],
                                  stride=1, padding='SAME'):
                # mixed: 100 x 140 x 64.
                with tf.variable_scope('mixed_100x146x64a'):
                    with tf.variable_scope('branch2x2'):
                        branch2x2 = ops.conv2d(net, 16, [2, 2], stride=2, prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.conv2d(branch2x2, 16, [3, 3], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(branch3x3, 64, [1, 1], batch_norm_params=True)
                    with tf.variable_scope('branch_dropout'):
                        branch_dropout = ops.dropout(branch1x1, 0.01, scope='dropout')
                    with tf.variable_scope('branch_pool'):
                        branch_pool_temp = ops.max_pool(net, [2, 2], stride=2, padding='SAME')
                        N, H, W, C = branch_pool_temp.get_shape()
                        branch_pool = tf.ones([N, H, W, 64 - 16])
                        branch_pool = tf.concat(3, [branch_pool_temp, branch_pool])
                    with tf.variable_scope('branch_pool_dropout'):
                        branch_pool_dropout = tf.add(branch_pool, branch_dropout)
                    net = ops.prelu(branch_pool_dropout)
                    end_points['mixed_35x35x256a'] = net

                # mixed: 100 x 140 x 64.
                with tf.variable_scope('mixed_100x146x64b'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 16, [1, 1], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.conv2d(branch1x1, 16, [3, 3], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(branch3x3, 64, [1, 1], batch_norm_params=True)
                    with tf.variable_scope('branch_dropout'):
                        branch_dropout = ops.dropout(branch1x1, 0.01, scope='dropout')
                        branch_dropout = tf.add(net, branch_dropout)
                    net = ops.prelu(branch_dropout)
                    end_points['mixed_35x35x256b'] = net

                # mixed: 100 x 140 x 64.
                with tf.variable_scope('mixed_100x146x64c'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 16, [1, 1], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.conv2d(branch1x1, 16, [3, 3], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(branch3x3, 64, [1, 1], batch_norm_params=True)
                    with tf.variable_scope('branch_dropout'):
                        branch_dropout = ops.dropout(branch1x1, 0.01, scope='dropout')
                        branch_dropout = tf.add(net, branch_dropout)
                    net = ops.prelu(branch_dropout)
                    end_points['mixed_35x35x256c'] = net

                # mixed: 100 x 140 x 64.
                with tf.variable_scope('mixed_100x146x64d'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 16, [1, 1], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.conv2d(branch1x1, 16, [3, 3], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(branch3x3, 64, [1, 1], batch_norm_params=True)
                    with tf.variable_scope('branch_dropout'):
                        branch_dropout = ops.dropout(branch1x1, 0.01, scope='dropout')
                        branch_dropout = tf.add(net, branch_dropout)
                    net = ops.prelu(branch_dropout)
                    end_points['mixed_35x35x256d'] = net

                # mixed: 100 x 140 x 64.
                with tf.variable_scope('mixed_100x146x64e'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 16, [1, 1], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.conv2d(branch1x1, 16, [3, 3], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(branch3x3, 64, [1, 1], batch_norm_params=True)
                    with tf.variable_scope('branch_dropout'):
                        branch_dropout = ops.dropout(branch1x1, 0.01, scope='dropout')
                        branch_dropout = tf.add(net, branch_dropout)
                    net = ops.prelu(branch_dropout)
                    end_points['mixed_35x35x256e'] = net

                # mixed_1: 50 x 70 x 128.
                with tf.variable_scope('mixed_50x70x128a'):
                    with tf.variable_scope('branch2x2'):
                        branch2x2 = ops.conv2d(net, 32, [2, 2], stride=2, prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.conv2d(branch2x2, 32, [3, 3], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(branch3x3, 128, [1, 1], batch_norm_params=True)
                    with tf.variable_scope('branch_dropout'):
                        branch_dropout = ops.dropout(branch1x1, 0.1, scope='dropout')
                    with tf.variable_scope('branch_pool'):
                        branch_pool_temp = ops.max_pool(net, [2, 2], stride=2, padding='SAME')
                        N, H, W, C = branch_pool_temp.get_shape()
                        branch_pool = tf.ones([N, H, W, 64 - 16])
                        branch_pool = tf.concat(3, [branch_pool_temp, branch_pool])
                    with tf.variable_scope('branch_pool_dropout'):
                        branch_pool_dropout = tf.add(branch_pool, branch_dropout)
                    net = ops.prelu(branch_pool_dropout)
                    end_points['mixed_50x70x128a'] = net

                # mixed_2: 50 x 70 x 128.
                with tf.variable_scope('mixed_50x70x128b'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 32, [1, 1], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.conv2d(branch1x1, 32, [3, 3], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(branch3x3, 128, [1, 1], batch_norm_params=True)
                    with tf.variable_scope('branch_dropout'):
                        branch_dropout = ops.dropout(branch1x1, 0.1, scope='dropout')
                        branch_dropout = tf.add(net, branch_dropout)
                    net = ops.prelu(branch_dropout)
                    end_points['mixed_50x70x128b'] = net

                # mixed_3: 50 x 70 x 128.
                with tf.variable_scope('mixed_50x70x128c'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 32, [1, 1], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.atrous_conv2d(branch1x1, 32, [3, 3], rate=2, prelu_activation=True,
                                                      batch_norm_params=True)
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(branch3x3, 128, [1, 1], batch_norm_params=True)
                    with tf.variable_scope('branch_dropout'):
                        branch_dropout = ops.dropout(branch1x1, 0.1, scope='dropout')
                        branch_dropout = tf.add(net, branch_dropout)
                    net = ops.prelu(branch_dropout)
                    end_points['mixed_50x70x128c'] = net

                # mixed_4: 50 x 70 x 128.
                with tf.variable_scope('mixed_50x70x128d'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 32, [1, 1], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.conv2d(branch1x1, 32, [5, 1])
                        branch3x3 = ops.conv2d(branch3x3, 32, [1, 5], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(branch3x3, 128, [1, 1], batch_norm_params=True)
                    with tf.variable_scope('branch_dropout'):
                        branch_dropout = ops.dropout(branch1x1, 0.1, scope='dropout')
                        branch_dropout = tf.add(net, branch_dropout)
                    net = ops.prelu(branch_dropout)
                    end_points['mixed_50x70x128d'] = net

                # mixed_5: 50 x 70 x 128.
                with tf.variable_scope('mixed_50x70x128e'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 32, [1, 1], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.atrous_conv2d(branch1x1, 32, [3, 3], rate=4, prelu_activation=True,
                                                      batch_norm_params=True)
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(branch3x3, 128, [1, 1], batch_norm_params=True)
                    with tf.variable_scope('branch_dropout'):
                        branch_dropout = ops.dropout(branch1x1, 0.1, scope='dropout')
                        branch_dropout = tf.add(net, branch_dropout)
                    net = ops.prelu(branch_dropout)
                    end_points['mixed_50x70x128e'] = net

                # mixed_6: 50 x 70 x 128.
                with tf.variable_scope('mixed_50x70x128f'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 32, [1, 1], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.conv2d(branch1x1, 32, [3, 3], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(branch3x3, 128, [1, 1], batch_norm_params=True)
                    with tf.variable_scope('branch_dropout'):
                        branch_dropout = ops.dropout(branch1x1, 0.1, scope='dropout')
                        branch_dropout = tf.add(net, branch_dropout)
                    net = ops.prelu(branch_dropout)
                    end_points['mixed_50x70x128f'] = net

                # mixed_7: 50 x 70 x 128.
                with tf.variable_scope('mixed_50x70x128g'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 32, [1, 1], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.atrous_conv2d(branch1x1, 32, [3, 3], rate=8, prelu_activation=True,
                                                      batch_norm_params=True)
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(branch3x3, 128, [1, 1], batch_norm_params=True)
                    with tf.variable_scope('branch_dropout'):
                        branch_dropout = ops.dropout(branch1x1, 0.1, scope='dropout')
                        branch_dropout = tf.add(net, branch_dropout)
                    net = ops.prelu(branch_dropout)
                    end_points['mixed_50x70x128g'] = net

                # mixed_8: 50 x 70 x 128.
                with tf.variable_scope('mixed_50x70x128h'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 32, [1, 1], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.conv2d(branch1x1, 32, [5, 1])
                        branch3x3 = ops.conv2d(branch3x3, 32, [1, 5], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(branch3x3, 128, [1, 1], batch_norm_params=True)
                    with tf.variable_scope('branch_dropout'):
                        branch_dropout = ops.dropout(branch1x1, 0.1, scope='dropout')
                        branch_dropout = tf.add(net, branch_dropout)
                    net = ops.prelu(branch_dropout)
                    end_points['mixed_50x70x128h'] = net

                # mixed_9: 50 x 70 x 128.
                with tf.variable_scope('mixed_50x70x128i'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 32, [1, 1], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.atrous_conv2d(branch1x1, 32, [3, 3], rate=16, prelu_activation=True,
                                                      batch_norm_params=True)
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(branch3x3, 128, [1, 1], batch_norm_params=True)
                    with tf.variable_scope('branch_dropout'):
                        branch_dropout = ops.dropout(branch1x1, 0.1, scope='dropout')
                        branch_dropout = tf.add(net, branch_dropout)
                    net = ops.prelu(branch_dropout)
                    end_points['mixed_50x70x128i'] = net

                # mixed_10: 50 x 70 x 128.
                with tf.variable_scope('mixed_50x70x128j'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 32, [1, 1], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.conv2d(branch1x1, 32, [3, 3], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(branch3x3, 128, [1, 1], batch_norm_params=True)
                    with tf.variable_scope('branch_dropout'):
                        branch_dropout = ops.dropout(branch1x1, 0.1, scope='dropout')
                        branch_dropout = tf.add(net, branch_dropout)
                    net = ops.prelu(branch_dropout)
                    end_points['mixed_50x70x128j'] = net

                # mixed_11: 50 x 70 x 128.
                with tf.variable_scope('mixed_50x70x128k'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 32, [1, 1], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.atrous_conv2d(branch1x1, 32, [3, 3], rate=2, prelu_activation=True,
                                                      batch_norm_params=True)
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(branch3x3, 128, [1, 1], batch_norm_params=True)
                    with tf.variable_scope('branch_dropout'):
                        branch_dropout = ops.dropout(branch1x1, 0.1, scope='dropout')
                        branch_dropout = tf.add(net, branch_dropout)
                    net = ops.prelu(branch_dropout)
                    end_points['mixed_50x70x128k'] = net

                # mixed_12: 50 x 70 x 128.
                with tf.variable_scope('mixed_50x70x128l'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 32, [1, 1], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.conv2d(branch1x1, 32, [5, 1])
                        branch3x3 = ops.conv2d(branch3x3, 32, [1, 5], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(branch3x3, 128, [1, 1], batch_norm_params=True)
                    with tf.variable_scope('branch_dropout'):
                        branch_dropout = ops.dropout(branch1x1, 0.1, scope='dropout')
                        branch_dropout = tf.add(net, branch_dropout)
                    net = ops.prelu(branch_dropout)
                    end_points['mixed_50x70x128l'] = net

                # mixed_13: 50 x 70 x 128.
                with tf.variable_scope('mixed_50x70x128m'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 32, [1, 1], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.atrous_conv2d(branch1x1, 32, [3, 3], rate=4, prelu_activation=True,
                                                      batch_norm_params=True)
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(branch3x3, 128, [1, 1], batch_norm_params=True)
                    with tf.variable_scope('branch_dropout'):
                        branch_dropout = ops.dropout(branch1x1, 0.1, scope='dropout')
                        branch_dropout = tf.add(net, branch_dropout)
                    net = ops.prelu(branch_dropout)
                    end_points['mixed_50x70x128m'] = net

                # mixed_14: 50 x 70 x 128.
                with tf.variable_scope('mixed_50x70x128n'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 32, [1, 1], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.conv2d(branch1x1, 32, [3, 3], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(branch3x3, 128, [1, 1], batch_norm_params=True)
                    with tf.variable_scope('branch_dropout'):
                        branch_dropout = ops.dropout(branch1x1, 0.1, scope='dropout')
                        branch_dropout = tf.add(net, branch_dropout)
                    net = ops.prelu(branch_dropout)
                    end_points['mixed_50x70x128n'] = net

                # mixed_15: 50 x 70 x 128.
                with tf.variable_scope('mixed_50x70x128o'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 32, [1, 1], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.atrous_conv2d(branch1x1, 32, [3, 3], rate=8, prelu_activation=True,
                                                      batch_norm_params=True)
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(branch3x3, 128, [1, 1], batch_norm_params=True)
                    with tf.variable_scope('branch_dropout'):
                        branch_dropout = ops.dropout(branch1x1, 0.1, scope='dropout')
                        branch_dropout = tf.add(net, branch_dropout)
                    net = ops.prelu(branch_dropout)
                    end_points['mixed_50x70x128o'] = net

                # mixed_17: 50 x 70 x 128.
                with tf.variable_scope('mixed_50x70x128p'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 32, [1, 1], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.conv2d(branch1x1, 32, [5, 1])
                        branch3x3 = ops.conv2d(branch3x3, 32, [1, 5], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(branch3x3, 128, [1, 1], batch_norm_params=True)
                    with tf.variable_scope('branch_dropout'):
                        branch_dropout = ops.dropout(branch1x1, 0.1, scope='dropout')
                        branch_dropout = tf.add(net, branch_dropout)
                    net = ops.prelu(branch_dropout)
                    end_points['mixed_50x70x128p'] = net

                # mixed_18: 50 x 70 x 128.
                with tf.variable_scope('mixed_50x70x128q'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 32, [1, 1], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.atrous_conv2d(branch1x1, 32, [3, 3], rate=16, prelu_activation=True,
                                                      batch_norm_params=True)
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(branch3x3, 128, [1, 1], batch_norm_params=True)
                    with tf.variable_scope('branch_dropout'):
                        branch_dropout = ops.dropout(branch1x1, 0.1, scope='dropout')
                        branch_dropout = tf.add(net, branch_dropout)
                    net = ops.prelu(branch_dropout)
                    end_points['mixed_50x70x128q'] = net

                # mixed_19: 100 x 140 x 64.
                with tf.variable_scope('mixed_100x140x64r'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 16, [1, 1], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.conv2d_transpose(branch1x1, 16, [3, 3], stride=2, prelu_activation=True,
                                                         batch_norm_params=True)
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(branch3x3, 64, [1, 1], batch_norm_params=True)
                    net = branch1x1
                    end_points['mixed_100x140x64r'] = net

                # mixed_20: 100 x 140 x 64.
                with tf.variable_scope('mixed_100x140x64s'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 16, [1, 1], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.conv2d(branch1x1, 16, [3, 3], prelu_activation=True,
                                               batch_norm_params=True)
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(branch3x3, 64, [1, 1], batch_norm_params=True)
                    net = branch1x1
                    end_points['mixed_100x140x64s'] = net

                # mixed_21: 100 x 140 x 64.
                with tf.variable_scope('mixed_100x140x64t'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 16, [1, 1], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.conv2d(branch1x1, 16, [3, 3], prelu_activation=True,
                                               batch_norm_params=True)
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(branch3x3, 64, [1, 1], batch_norm_params=True)
                    net = branch1x1
                    end_points['mixed_100x140x64t'] = net

                # mixed_22: 200 x 280 x 16.
                with tf.variable_scope('mixed_200x280x16u'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 4, [1, 1], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.conv2d_transpose(branch1x1, 4, [3, 3], stride=2, prelu_activation=True,
                                                         batch_norm_params=True)
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(branch3x3, 16, [1, 1], batch_norm_params=True)
                    net = branch1x1
                    end_points['mixed_200x280x16u'] = net

                # mixed_23: 200 x 280 x 16.
                with tf.variable_scope('mixed_200x280x16v'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(net, 4, [1, 1], prelu_activation=True, batch_norm_params=True)
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = ops.conv2d(branch1x1, 4, [3, 3], prelu_activation=True,
                                               batch_norm_params=True)
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = ops.conv2d(branch3x3, 16, [1, 1], batch_norm_params=True)
                    net = branch1x1
                    end_points['mixed_200x280x16v'] = net

                # Final pooling and prediction
                # 400 x 560 x 2
                with tf.variable_scope('logits'):
                    logits = ops.conv2d_transpose(net, num_classes, [2, 2], stride=2, prelu_activation=True,
                                                  batch_norm_params=True)
                    end_points['logits'] = logits
                    end_points['predictions'] = tf.argmax(logits, 3, name='predictions')
            return logits, end_points


def inception_v3_parameters(weight_decay=0.00004, stddev=0.1,
                            batch_norm_decay=0.9997, batch_norm_epsilon=0.001):
    """Yields the scope with the default parameters for inception_v3.

    Args:
      weight_decay: the weight decay for weights variables.
      stddev: standard deviation of the truncated guassian weight distribution.
      batch_norm_decay: decay for the moving average of batch_norm momentums.
      batch_norm_epsilon: small float added to variance to avoid dividing by zero.

    Yields:
      a arg_scope with the parameters needed for inception_v3.
    """
    # Set weight_decay for weights in Conv and FC layers.
    with scopes.arg_scope([ops.conv2d, ops.fc],
                          weight_decay=weight_decay):
        # Set stddev, activation and parameters for batch_norm.
        with scopes.arg_scope([ops.conv2d],
                              stddev=stddev,
                              activation=tf.nn.relu,
                              batch_norm_params={
                                  'decay': batch_norm_decay,
                                  'epsilon': batch_norm_epsilon}) as arg_scope:
            yield arg_scope
