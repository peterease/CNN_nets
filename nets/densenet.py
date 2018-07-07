"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
    current = slim.dropout(current, scope=scope + '_dropout')
    return current


def block(net, layers, growth, scope='block'):
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1' + str(idx))
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
    return net


def densenet(images, num_classes=1001, is_training=False,
             dropout_keep_prob=0.8,
             scope='densenet'):
    """Creates a variant of the densenet model.

      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    growth = 24
    compression_rate = 0.5

    def reduce_dim(input_feature):
        return int(int(input_feature.shape[-1]) * compression_rate)

    end_points = {}

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,
                                         keep_prob=dropout_keep_prob)) as ssc:
            
            pass
            ##########################
            # Put your code here.
            
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride = 1, padding = "SAME"):                
                def transition_layer(net, out_nums, scope = "transition_layer"):
                    net = slim.conv2d(net, out_nums, [1,1], scope = scope + "_Conv2d_1x1")
                    net = slim.avg_pool2d(net, [2,2], stride = 2, scope = scope + "_AvgPool_3x3")
                    return net
                
                end_point = "Pre_block_layer"
                #224 x 224 x3
                net = slim.conv2d(images, 2*growth, [7,7], stride = 2, scope = end_point + "_Conv2d_7x7")
                net = slim.max_pool2d(net, [3,3], stride =2, scope = end_point + "_Maxpool_3x3")
                end_points[end_point] = net
                
                #56 x 56 x 48 
                end_point = "Block_1"
                net = block(net, 6, growth, scope = end_point) #56 x 56 x 8*growth
                end_points[end_point] = net
                
                
                #56 x 56 x 192
                end_point = "Transition_1"
                net = transition_layer(net, reduce_dim(net), scope = end_point)
                end_points[end_point] = net
                
                #28 x 28 x 96
                end_point = "Block_2"
                net = block(net, 12, growth, scope = end_point) #28 x 28 x 16*growth
                end_points[end_point] = net
                
                #28 x 28 x 384
                end_point = "Transition_2"
                net = transition_layer(net, reduce_dim(net), scope = end_point)
                end_points[end_point] = net
                
                #14 x 14 x 192
                end_point = "Block_3"
                net = block(net, 32, growth, scope = end_point) #14 x 14 x 40*growth
                end_points[end_point] = net
                
                #14 x 14 x 960
                end_point = "Transition_3"
                net = transition_layer(net, reduce_dim(net), scope = end_point)
                end_points[end_point] = net
                
                #7 x 7 x 480
                end_point = "Block_4"
                net = block(net, 32, growth, scope = end_point) #14 x 14 x 32*growth #7 x 7 x 52*growth
                end_points[end_point] = net
                
                #7 x 7 x 1248                
                net = slim.avg_pool2d(net, [7,7], padding = "VALID", scope = "Pre_logits")
                with slim.arg_scope(densenet_arg_scope()):                    
                    logits = slim.conv2d(net, num_classes, [1,1])
                    logits = tf.squeeze(logits, [1,2], scope = "Logits")
                end_points["Logits"] = logits
                end_points["Predictions"] = slim.softmax(logits, scope = "Predictions") 
                
            ##########################
    return logits, end_points


def bn_drp_scope(is_training=True, keep_prob=0.8):
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope(
        [slim.batch_norm],
            scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope(
            [slim.dropout],
                is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc


def densenet_arg_scope(weight_decay=0.004):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False),
        activation_fn=None, biases_initializer=None, padding='SAME',
            stride=1) as sc:
        return sc


densenet.default_image_size = 224
