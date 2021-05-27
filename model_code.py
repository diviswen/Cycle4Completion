import numpy as np
import tensorflow as tf
import net_util as nu



def placeholder_inputs(batch_size, num_point, num_point_gt):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    pointclouds_Y = tf.placeholder(tf.float32, shape=(batch_size, num_point_gt, 3))
    pointclouds_gt = tf.placeholder(tf.float32, shape=(batch_size, num_point_gt, 3))
    is_training = tf.placeholder(tf.bool,shape=[])
    return pointclouds_pl, pointclouds_Y, pointclouds_gt, is_training


def get_model(X_inputs, Y_inputs, is_training, bn_decay=None, weight_decay=None):
    """
    Args:
        point_clouds: (batch_size, num_point, 3)
    Returns:
        pointclouds_pred: (batch_size, num_point, 3)
    """
    batch_size = X_inputs.get_shape()[0].value
    num_point = X_inputs.get_shape()[1].value
    nu.args.phase = is_training
    print(is_training)

    X_feats = nu.create_pcn_encoder(X_inputs, name='X')
    pred_X = nu.create_decoder(X_feats, name='X')
    X2Y_feats, X2Y_code = nu.create_transferer_X2Y(X_feats, name='X2Y')
    print(X2Y_feats)
    print(X2Y_code)
    X2Y2X_feats, _ = nu.create_transferer_Y2X(X2Y_feats, X2Y_code, name='Y2X')
    pred_X2Y2X = nu.create_decoder(X2Y2X_feats, name='X')
    complete_X = nu.create_decoder(X2Y_feats, name='Y')
    

    Y_feats = nu.create_pcn_encoder(Y_inputs, name='Y')
    pred_Y = nu.create_decoder(Y_feats, name='Y')
    Y2X_feats, Y2X_code = nu.create_transferer_Y2X(Y_feats, None, name='Y2X')
    Y2X2Y_feats, Y2X2Y_code = nu.create_transferer_X2Y(Y2X_feats, name='X2Y')
    pred_Y2X2Y = nu.create_decoder(Y2X2Y_feats, name='Y')
    incomplete_Y = nu.create_decoder(Y2X_feats, name='X')

    X2Y_logits = nu.create_discrminator(X2Y_feats, name='Y')
    Y_logits = nu.create_discrminator(Y_feats, name='Y')

    Y2X_logits = nu.create_discrminator(Y2X_feats, name='X')
    X_logits = nu.create_discrminator(X_feats, name='X')

    return pred_X, pred_Y, pred_Y2X2Y, pred_X2Y2X, X2Y_logits, Y_logits, Y2X_logits, X_logits,\
     X_feats, X2Y_feats, Y_feats, Y2X_feats, complete_X, incomplete_Y, Y2X2Y_feats, X2Y2X_feats, X2Y_code, Y2X_code, Y2X2Y_code

def get_loss(pred_X, pred_Y, pred_Y2X2Y, pred_X2Y2X, X2Y_logits, Y_logits, Y2X_logits, X_logits, X_feats, X2Y_feats,\
             Y_feats, Y2X_feats, complete_X, incomplete_Y, gt_X, gt_Y, gt_GT, Y2X2Y_feats, X2Y2X_feats, X2Y_code, Y2X_code, Y2X2Y_code):

    batch_size = gt_X.get_shape()[0].value#

    complete_CD = 2048*nu.chamfer(complete_X, gt_GT)
    chamfer_loss_X_cycle = 2048 * nu.chamfer(pred_X2Y2X, gt_X)
    chamfer_loss_Y_cycle = 2048 * nu.chamfer(pred_Y2X2Y, gt_Y)

    chamfer_loss_partial_X2Y = 2048 * nu.chamfer_single_side(gt_X, complete_X)
    chamfer_loss_partial_Y2X = 2048 * nu.chamfer_single_side(incomplete_Y, gt_Y)
    

    #optimizing encoder and decoder
    chamfer_loss_X = 2048 * nu.chamfer(pred_X, gt_X)
    chamfer_loss_Y = 2048 * nu.chamfer(pred_Y, gt_Y)
    

    #optimizing discrminator
    D_loss_X = X_logits - Y2X_logits
    D_loss_Y = Y_logits - X2Y_logits
    

    epsilon = tf.random_uniform([], 0.0, 1.0)

    x_hat = epsilon*X_feats +(1-epsilon)*Y2X_feats
    d_hat = nu.create_discrminator(x_hat, name='X')
    gradients = tf.gradients(d_hat, [x_hat])[0]

    gradients = tf.reshape(gradients, shape=[batch_size, -1])
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
    gp_X = tf.reduce_mean(tf.square(slopes - 1)*10)

    y_hat = epsilon*Y_feats +(1-epsilon)*X2Y_feats
    d_hat = nu.create_discrminator(y_hat, name='Y')
    gradients = tf.gradients(d_hat, [y_hat])[0]
    gradients = tf.reshape(gradients, shape=[batch_size, -1])
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
    gp_Y = tf.reduce_mean(tf.square(slopes - 1)*10)

    D_loss = D_loss_Y + D_loss_X + tf.minimum((gp_Y + gp_X),10e7)

    #optimizing transferer
    G_loss_X2Y = -D_loss_Y
    G_loss_Y2X = -D_loss_X

    code_loss = tf.reduce_mean(tf.square(Y2X_code - Y2X2Y_code))*100

    ED_loss = chamfer_loss_X + chamfer_loss_Y
    Trans_loss = (G_loss_X2Y + G_loss_Y2X)*.1 + (chamfer_loss_partial_X2Y + chamfer_loss_partial_Y2X)*1.0  + (chamfer_loss_Y_cycle + chamfer_loss_X_cycle)*0.01 + code_loss

    return ED_loss, Trans_loss, D_loss, chamfer_loss_X, chamfer_loss_Y, chamfer_loss_X_cycle, chamfer_loss_Y_cycle,\
            D_loss_X, D_loss_Y, complete_CD, chamfer_loss_partial_X2Y, chamfer_loss_partial_Y2X, code_loss

