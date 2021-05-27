import numpy as np
import tensorflow as tf
import math
from externals.structural_losses import tf_nndistance, tf_approxmatch, tf_hausdorff_distance

tree_arch = {}
tree_arch[2] = [32, 64]
tree_arch[4] = [4, 8, 8, 8]
tree_arch[6] = [2, 4, 4, 4, 4, 4]
tree_arch[8] = [2, 2, 2, 2, 2, 4, 4, 4]

def get_arch(nlevels, npts):
    #logmult = int(math.log2(npts/2048))
    logmult = int(math.log(npts/2048, 2))
    assert 2048*(2**(logmult)) == npts, "Number of points is %d, expected 2048x(2^n)" % (npts)
    arch = tree_arch[nlevels]
    while logmult > 0:
        last_min_pos = np.where(arch==np.min(arch))[0][-1]
        arch[last_min_pos]*=2
        logmult -= 1
    return arch

class TopnetFlag(object):
    def __init__(self):
        self.ENCODER_ID = 1 # 0 for pointnet encoder & 1 for pcn encoder
        self.phase = None
        self.code_nfts = 1024
        self.npts = 2048
        self.NFEAT = 8
        self.NLEVELS = 6
        self.tarch = get_arch(self.NLEVELS, self.npts)
args = TopnetFlag()


def create_discrminator(inputs, name=''):
    with tf.variable_scope('discriminator_%s'%(name), reuse=tf.AUTO_REUSE):
        inputs = mlp(inputs, [512,256,128,1], args.phase, bn=False)
        return tf.reduce_mean(inputs)

def create_transferer_X2Y(inputs, name='X2Y'):
    with tf.variable_scope('transferer_%s'%(name), reuse=tf.AUTO_REUSE):
        inputs = tf.expand_dims(inputs, axis=1)
        inputs = mlp_conv(inputs, [1024, 1024, 1024, 1024, 1024+2], args.phase)
        inputs = tf.squeeze(inputs)
        codeword = inputs[:,1024:1026]
        codeword = tf.sigmoid(codeword)
        inputs = inputs[:,0:1024]
        return inputs, codeword

def create_transferer_Y2X(inputs, codeword=None, name='Y2X'):
    if codeword is None:
        codeword = tf.random_uniform([inputs.shape[0].value, 2], maxval=.5)
    with tf.variable_scope('transferer_%s'%(name), reuse=tf.AUTO_REUSE):
        inputs = tf.expand_dims(tf.concat([inputs,codeword],axis=-1), axis=1)
        print(inputs)
        inputs = mlp_conv(inputs, [1024, 1024, 1024, 1024, 1024], args.phase)
        inputs = tf.squeeze(inputs)
        return inputs, codeword

def create_pcn_encoder(inputs, name=''):
    with tf.variable_scope('encoder_0_%s'%(name), reuse=tf.AUTO_REUSE):
        features = mlp_conv(inputs, [128, 256], args.phase)
        features_global = tf.reduce_max(features, axis=1, keep_dims=True, name='maxpool_0')
        features = tf.concat([features, tf.tile(features_global, [1, tf.shape(inputs)[1], 1])], axis=2)
    with tf.variable_scope('encoder_1_%s'%(name), reuse=tf.AUTO_REUSE):
        features = mlp_conv(features, [512, args.code_nfts], args.phase)
        features = tf.reduce_max(features, axis=1, name='maxpool_1')
    return features

def chamfer(pcd1, pcd2):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    mdist1 = tf.reduce_mean(dist1)
    mdist2 = tf.reduce_mean(dist2)
    return mdist1 + mdist2

def chamfer_single_side(pcd1, pcd2):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    mdist1 = tf.reduce_mean(dist1)
    return mdist1

def emd(pcd1, pcd2):
    num_points = tf.cast(pcd2.shape[1], tf.float32)
    match = tf_approxmatch.approx_match(pcd1, pcd2)
    cost = tf_approxmatch.match_cost(pcd1, pcd2, match)
    return cost / num_points

def mlp(features, layer_dims, phase, bn=None):
    for i, num_outputs in enumerate(layer_dims[:-1]):
        features = tf.contrib.layers.fully_connected(
            features, num_outputs,
            activation_fn=None,
            normalizer_fn=None,
            scope='fc_%d' % i)
        if bn:
            with tf.variable_scope('fc_bn_%d' % (i), reuse=tf.AUTO_REUSE):
                features = tf.layers.batch_normalization(features, training=phase)
        features = tf.nn.relu(features, 'fc_relu_%d' % i)

    outputs = tf.contrib.layers.fully_connected(
        features, layer_dims[-1],
        activation_fn=None,
        scope='fc_%d' % (len(layer_dims) - 1))
    return outputs


def mlp_conv(inputs, layer_dims, phase, bn=None):
    inputs = tf.expand_dims(inputs, 1)  
    for i, num_out_channel in enumerate(layer_dims[:-1]):
        inputs = tf.contrib.layers.conv2d(
            inputs, num_out_channel,
            kernel_size=[1, 1],
            activation_fn=None,
            normalizer_fn=None,
            scope='conv_%d' % i)
        if bn:
            with tf.variable_scope('conv_bn_%d' % (i), reuse=tf.AUTO_REUSE):
                inputs = tf.layers.batch_normalization(inputs, training=phase)
        inputs = tf.nn.relu(inputs, 'conv_relu_%d' % i)
    outputs = tf.contrib.layers.conv2d(
        inputs, layer_dims[-1],
        kernel_size=[1, 1],
        activation_fn=None,
        scope='conv_%d' % (len(layer_dims) - 1))
    outputs = tf.squeeze(outputs, [1])  # modified: conv1d -> conv2d
    return outputs

def create_level(level, input_channels, output_channels, inputs, bn):
    with tf.variable_scope('level_%d' % (level), reuse=tf.AUTO_REUSE):
        features = mlp_conv(inputs, [input_channels, int(input_channels/2),
                                        int(input_channels/4), int(input_channels/8),
                                        output_channels*int(args.tarch[level])],
                                    args.phase, bn)
        features = tf.reshape(features, [tf.shape(features)[0], -1, output_channels])
    return features

def create_decoder(code, name=''):
    Nin = args.NFEAT + args.code_nfts
    Nout = args.NFEAT
    bn = True
    N0 = int(args.tarch[0])
    nlevels = len(args.tarch)
    with tf.variable_scope('decoder_%s'%(name), reuse=tf.AUTO_REUSE):
        level0 = mlp(code, [256, 64, args.NFEAT * N0], args.phase, bn=True)
        level0 = tf.tanh(level0, name='tanh_0')
        level0 = tf.reshape(level0, [-1, N0, args.NFEAT])
        outs = [level0, ]
        for i in range(1, nlevels):
            if i == nlevels - 1:
                Nout = 3
                bn = False
            inp = outs[-1]
            y = tf.expand_dims(code, 1)
            y = tf.tile(y, [1, tf.shape(inp)[1], 1])
            y = tf.concat([inp, y], 2)
            outs.append(tf.tanh(create_level(i, Nin, Nout, y, bn), name='tanh_%d' % (i)))
            
    return outs[-1]