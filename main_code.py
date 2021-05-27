import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import time
import math
from datetime import datetime
import argparse
import importlib
import random
import numpy as np
import tensorflow as tf
import data_provider as dp
import io_util

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='model_code', help='Model name [default: model_l2h]')
parser.add_argument('--log_dir', default='logs', help='Log dir [default: logs]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [1024/2048] [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=500, help='Epoch to run [default: 400]')
parser.add_argument('--min_epoch', type=int, default=0, help='Epoch from which training starts [default: 0]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--decay_step', type=int, default=400000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--weight_decay', type=float, default=0.000010, help='Weight decay [default: 0.007]')
parser.add_argument('--warmup_step', type=int, default=10, help='Warm up step for lr [default: 200000]')
parser.add_argument('--gamma_cd', type=float, default=10.0, help='Gamma for chamfer loss [default: 10.0]')
parser.add_argument('--restore', default='None', help='Restore path [default: None]')
parser.add_argument('--data_category', default='car', help='plane/car/lamp/chair/table/cabinet/watercraft/sofa')


FLAGS = parser.parse_args()
BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
MIN_EPOCH = FLAGS.min_epoch
NUM_POINT = FLAGS.num_point
NUM_POINT_GT = FLAGS.num_point
BASE_LEARNING_RATE = FLAGS.learning_rate
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
WEIGHT_DECAY = FLAGS.weight_decay
DATA_CATEGORY = FLAGS.data_category
if WEIGHT_DECAY <= 0.:
    WEIGHT_DECAY = None
WARMUP_STEP = float(FLAGS.warmup_step)
GAMMA_CD = FLAGS.gamma_cd
MODEL = importlib.import_module(FLAGS.model)
MODEL_FILE = FLAGS.model
TIME = time.strftime("%m%d-%H%M%S", time.localtime())
MODEL_NAME = '%s_%s' % (FLAGS.model, TIME)

LOG_DIR = os.path.join(FLAGS.log_dir, DATA_CATEGORY+MODEL_NAME)
RESTORE_PATH = FLAGS.restore


BN_INIT_DECAY = 0.1
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99


if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    os.makedirs(os.path.join(LOG_DIR,'vis'))
os.system('cp %s.py %s/%s_%s.py' % (MODEL_FILE, LOG_DIR, MODEL_FILE, TIME))
os.system('cp main_code.py %s/main_code_%s.py' % (LOG_DIR,TIME))


LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_RESULT_FOUT = open(os.path.join(LOG_DIR, 'log_result_%s.csv'%(TIME)), 'w')
LOG_RESULT_FOUT.write('total_loss,emd_loss,repulsion_loss,chamfer_loss,l2_reg_loss,lr\n')


encode = {
    "chair": "03001627",
    "table": "04379243",
    "sofa": "04256520",
    "cabinet": "02933112",
    "lamp": "03636649",
    "car": "02958343",
    "plane": "02691156",
    "watercraft": "04530566"
}

DATA_PATH = os.path.join('./dataset/3depn', DATA_CATEGORY)
TRAIN_DATASET, TRAIN_DATASET_GT, TEST_DATASET, TEST_DATASET_GT = dp.load_completion_data(DATA_PATH, BATCH_SIZE, encode[DATA_CATEGORY], npoint=NUM_POINT, split='split_pcl2pcl.txt')


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


log_string(str(FLAGS))
log_string('TRAIN_DATASET: ' + str(TRAIN_DATASET.shape))
log_string('TEST_DATASET: ' + str(TEST_DATASET.shape))


def shuffle_dataset():
    data = np.reshape(TRAIN_DATASET, [-1, NUM_POINT, 3])
    gt = np.reshape(TRAIN_DATASET_GT, [-1, NUM_POINT, 3])
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    data = data[idx, ...]
    gt = gt[idx, ...]
    return np.reshape(data, (-1, BATCH_SIZE, NUM_POINT, 3)), np.reshape(gt, (-1, BATCH_SIZE, NUM_POINT, 3))



def get_learning_rate(batch):
    lr_wu = batch * BATCH_SIZE / WARMUP_STEP * BASE_LEARNING_RATE
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE / DECAY_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.minimum(learning_rate, lr_wu)
    learning_rate = tf.maximum(learning_rate, 0.000001) # CLIP THE LEARNING RATE!
    return learning_rate        


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            pointclouds_pl, pointclouds_Y, pointclouds_gt, is_training = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_POINT_GT)

            batch = tf.get_variable('batch', [], initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            pred_X, pred_Y, pred_Y2X2Y, pred_X2Y2X, X2Y_logits, Y_logits, Y2X_logits, X_logits, X_feats, X2Y_feats,\
             Y_feats, Y2X_feats, complete_X, incomplete_Y, Y2X2Y_feats, X2Y2X_feats, X2Y_code, Y2X_code, Y2X2Y_code = \
                    MODEL.get_model(pointclouds_pl, pointclouds_Y, is_training, bn_decay, WEIGHT_DECAY)
            ED_loss, Trans_loss, D_loss, chamfer_loss_X, chamfer_loss_Y, chamfer_loss_X_cycle, chamfer_loss_Y_cycle, D_loss_X, D_loss_Y,\
            complete_CD, chamfer_loss_partial_X2Y, chamfer_loss_partial_Y2X, code_loss = \
                    MODEL.get_loss(pred_X, pred_Y, pred_Y2X2Y, pred_X2Y2X, X2Y_logits, Y_logits, Y2X_logits, X_logits, X_feats, X2Y_feats, Y_feats, \
                        Y2X_feats, complete_X, incomplete_Y, pointclouds_pl, pointclouds_Y, pointclouds_gt, Y2X2Y_feats, X2Y2X_feats, X2Y_code, Y2X_code, Y2X2Y_code)
            
            tf.summary.scalar('chamfer_loss_X', chamfer_loss_X)
            tf.summary.scalar('chamfer_loss_Y', chamfer_loss_Y)
            tf.summary.scalar('chamfer_loss_X_cycle', chamfer_loss_X_cycle)
            tf.summary.scalar('chamfer_loss_Y_cycle', chamfer_loss_Y_cycle)
            tf.summary.scalar('complete_CD', complete_CD)
            tf.summary.scalar('D_loss_X', D_loss_X)
            tf.summary.scalar('D_loss_Y', D_loss_Y)
            tf.summary.scalar('chamfer_loss_partial_X2Y', chamfer_loss_partial_X2Y)
            

            var_list = tf.trainable_variables()
            ED_var = [var for var in var_list if ('encoder' in var.name) or ('decoder' in var.name)]
            Trans_var = [var for var in var_list if ('transferer' in var.name)]
            D_var = [var for var in var_list if 'discriminator' in var.name]

            ED_gradients = tf.gradients(ED_loss, ED_var)
            Trans_gradients = tf.gradients(Trans_loss, Trans_var)
            D_gradients = tf.gradients(D_loss, D_var)

            ED_g_and_v = zip(ED_gradients, ED_var)
            Trans_g_and_v = zip(Trans_gradients, Trans_var)
            D_g_and_v = zip(D_gradients, D_var)

            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)

            optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9)
            optimizer_D = tf.train.AdamOptimizer(BASE_LEARNING_RATE, beta1=0.9)
            optimizer_T = tf.train.AdamOptimizer(BASE_LEARNING_RATE)
            updata_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(updata_ops):
                ED_op = optimizer.apply_gradients(ED_g_and_v, global_step=batch)
                Trans_op = optimizer.apply_gradients(Trans_g_and_v, global_step=batch)

                D_op = optimizer_D.apply_gradients(D_g_and_v, global_step=batch)


            saver = tf.train.Saver(max_to_keep=300)
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        ckpt_state = tf.train.get_checkpoint_state(RESTORE_PATH)
        if ckpt_state is not None:
            LOAD_MODEL_FILE = os.path.join(RESTORE_PATH, os.path.basename(ckpt_state.model_checkpoint_path))
            saver.restore(sess, LOAD_MODEL_FILE)
            log_string('Model loaded in file: %s' % LOAD_MODEL_FILE)
        else:
            log_string('Failed to load model file: %s' % RESTORE_PATH)
            init = tf.global_variables_initializer()
            sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'pointclouds_Y': pointclouds_Y,
               'pointclouds_gt': pointclouds_gt,
               'is_training': is_training,
               'pointclouds_pred': complete_X,
               'incomplete_Y': incomplete_Y,
               'pred_Y2X2Y': pred_Y2X2Y,
               'pred_X2Y2X': pred_X2Y2X,
               'ED_loss': ED_loss,
               'code_loss': code_loss,
               'Trans_loss': Trans_loss,
               'D_loss': D_loss,
               'chamfer_loss_X': chamfer_loss_X,
               'chamfer_loss_Y': chamfer_loss_Y,
               'chamfer_loss_X_cycle': chamfer_loss_X_cycle,
               'chamfer_loss_Y_cycle': chamfer_loss_Y_cycle,
               'chamfer_loss_partial_X2Y': chamfer_loss_partial_X2Y,
               'chamfer_loss_partial_Y2X': chamfer_loss_partial_Y2X,
               'D_loss_X': D_loss_X,
               'D_loss_Y': D_loss_Y,
               'complete_CD': complete_CD,
               'learning_rate': learning_rate,
               'ED_op': ED_op,
               'Trans_op': Trans_op,
               'D_op': D_op,
               'step': batch,
               'merged': merged}
        min_emd = 999999.9
        min_cd = 999999.9
        min_emd_epoch = 0
        min_cd_epoch = 0
        for epoch in range(MIN_EPOCH, MAX_EPOCH):
            log_string('**** EPOCH %03d ****  \n%s' % (epoch, LOG_DIR))
            train_one_epoch(sess, ops, train_writer, epoch)
            cd_loss_i = eval_one_epoch(sess, ops, test_writer, epoch)
            if cd_loss_i < min_cd:
                min_cd = cd_loss_i
                min_cd_epoch = epoch
                save_path = saver.save(sess, os.path.join(LOG_DIR, 'checkpoints', 'min_cd.ckpt'))
                log_string('Model saved in file: %s' % save_path)
            log_string('min emd epoch: %d, emd = %f, min cd epoch: %d, cd = %f\n' % (min_emd_epoch, min_emd, min_cd_epoch, min_cd))
            

def train_one_epoch(sess, ops, train_writer, epoch):
    is_training = True
    log_string(str(datetime.now()))

    TRAIN_DATASET, TRAIN_DATASET_GT = shuffle_dataset()
    total_batch = TRAIN_DATASET.shape[0]
    ED_loss_sum = 0.
    Trans_loss_sum = 0.
    D_loss_sum = 0.
    chamfer_loss_X_sum = 0.
    chamfer_loss_Y_sum = 0.
    chamfer_loss_X_cycle_sum = 0.
    chamfer_loss_Y_cycle_sum = 0.
    D_loss_X_sum = 0.
    D_loss_Y_sum = 0.
    chamfer_loss_partial_X2Y_sum = 0.
    chamfer_loss_partial_Y2X_sum = 0.
    complete_CD_sum = 0.
    code_loss_sum = 0.

    for i in range(total_batch-2):
        batch_input_data = TRAIN_DATASET[i]
        batch_data_Y = TRAIN_DATASET_GT[i+1]
        batch_data_gt = TRAIN_DATASET_GT[i]

        feed_dict = {
            ops['pointclouds_pl']: batch_input_data[:, :, 0:3],
            ops['pointclouds_Y']: batch_data_Y[:, :, 0:3],
            ops['pointclouds_gt']: batch_data_gt[:, :, 0:3],
            ops['is_training']: is_training
        }

        summary, lr, step, ED_loss, Trans_loss, D_loss, chamfer_loss_X, chamfer_loss_Y, chamfer_loss_X_cycle, chamfer_loss_Y_cycle, \
        D_loss_X, D_loss_Y, complete_CD, chamfer_loss_partial_X2Y, chamfer_loss_partial_Y2X, code_loss, _, _, _ = \
        sess.run([ops['merged'],ops['learning_rate'], ops['step'], ops['ED_loss'],
            ops['Trans_loss'], ops['D_loss'], ops['chamfer_loss_X'], 
            ops['chamfer_loss_Y'], ops['chamfer_loss_X_cycle'], 
            ops['chamfer_loss_Y_cycle'], ops['D_loss_X'],
            ops['D_loss_Y'], ops['complete_CD'], ops['chamfer_loss_partial_X2Y'], ops['chamfer_loss_partial_Y2X'],
            ops['code_loss'],
            ops['ED_op'], ops['Trans_op'], ops['D_op']
            ], feed_dict=feed_dict)
        sess.run([ops['D_op']
            ], feed_dict=feed_dict)

        train_writer.add_summary(summary, step)
        ED_loss_sum += ED_loss
        Trans_loss_sum += Trans_loss
        D_loss_sum += D_loss
        code_loss_sum += code_loss
        chamfer_loss_X_sum += chamfer_loss_X
        chamfer_loss_Y_sum += chamfer_loss_Y
        chamfer_loss_X_cycle_sum += chamfer_loss_X_cycle
        chamfer_loss_Y_cycle_sum += chamfer_loss_Y_cycle
        D_loss_X_sum += D_loss_X
        D_loss_Y_sum += D_loss_Y
        complete_CD_sum += complete_CD
        chamfer_loss_partial_X2Y_sum += chamfer_loss_partial_X2Y
        chamfer_loss_partial_Y2X_sum += chamfer_loss_partial_Y2X

        k=10.
        if i%k==0:
            ED_loss_sum = ED_loss_sum/k
            Trans_loss_sum = Trans_loss_sum/k
            D_loss_sum = D_loss_sum/k
            chamfer_loss_X_sum = chamfer_loss_X_sum/k
            chamfer_loss_Y_sum = chamfer_loss_Y_sum/k
            chamfer_loss_X_cycle_sum = chamfer_loss_X_cycle_sum/k
            chamfer_loss_Y_cycle_sum = chamfer_loss_Y_cycle_sum/k
            D_loss_X_sum = D_loss_X_sum/k
            D_loss_Y_sum = D_loss_Y_sum/k
            complete_CD_sum = complete_CD_sum/k
            chamfer_loss_partial_X2Y_sum = chamfer_loss_partial_X2Y_sum/k
            chamfer_loss_partial_Y2X_sum = chamfer_loss_partial_Y2X_sum/k
            code_loss_sum = code_loss_sum/k

            print('%4d/%4d | ED: %.2f | Trans: %3.1f | D: %3.2f | X: %2.1f | Y: %2.1f | cycle_X: %.1f | cycle_Y: %.1f | WD_X: %3.1f | WD_Y: %3.1f | complete_CD: %3.1f | X2Y: %.1f | Y2X: %.1f | code: %.1f\n'
             % (i, total_batch,ED_loss_sum,Trans_loss_sum,D_loss_sum,chamfer_loss_X_sum*4.883,chamfer_loss_Y_sum*4.883,
                chamfer_loss_X_cycle_sum*4.883,chamfer_loss_Y_cycle_sum*4.883,
                D_loss_X_sum,D_loss_Y_sum,complete_CD_sum*4.883, chamfer_loss_partial_X2Y_sum*4.883, 
                chamfer_loss_partial_Y2X_sum*4.883, code_loss_sum)),
            ED_loss_sum = 0.
            Trans_loss_sum = 0.
            D_loss_sum = 0.
            chamfer_loss_X_sum = 0.
            chamfer_loss_Y_sum = 0.
            chamfer_loss_X_cycle_sum = 0.
            chamfer_loss_Y_cycle_sum = 0.
            D_loss_X_sum = 0.
            D_loss_Y_sum = 0.
            complete_CD_sum = 0.
            chamfer_loss_partial_X2Y_sum = 0.
            chamfer_loss_partial_Y2X_sum = 0.
            code_loss_sum = 0.

def eval_one_epoch(sess, ops, test_writer, epoch):
    is_training = False
    total_batch = TEST_DATASET.shape[0]
    chamfer_loss_sum = 0.


    for i in range(total_batch):
        batch_input_data = TEST_DATASET[i]
        batch_data_gt = TEST_DATASET_GT[i]

        feed_dict = {
            ops['pointclouds_pl']: batch_input_data[:, :, 0:3],
            ops['pointclouds_gt']: batch_data_gt[:, :, 0:3],
            ops['pointclouds_Y']: batch_data_gt[:, :, 0:3],
            ops['is_training']: is_training
        }
        complete_CD, pred_val, pred_Y2X, pred_Y2X2Y, pred_X2Y2X = sess.run([ops['complete_CD'], ops['pointclouds_pred'], ops['incomplete_Y'], ops['pred_Y2X2Y'],ops['pred_X2Y2X']], feed_dict=feed_dict)
        chamfer_loss_sum += complete_CD
  
    mean_chamfer_loss = chamfer_loss_sum / total_batch

    log_string('eval  chamfer loss: %.3f' % \
               (mean_chamfer_loss/2048. * 10000.))
    LOG_RESULT_FOUT.write('%.3f\n' % (mean_chamfer_loss/2048. * 10000.))
    LOG_RESULT_FOUT.flush()

    os.makedirs(os.path.join(LOG_DIR,'vis/epoch_%d_%.2f'%(epoch, mean_chamfer_loss*4.883)))
    for i in range(pred_val.shape[0]):
        gt = batch_data_gt[i]
        pred = pred_val[i]
        res = batch_input_data[i]
        Y2X = pred_Y2X[i]
        Y2X2Y = pred_Y2X2Y[i]
        X2Y2X = pred_X2Y2X[i]


        io_util.write_ply(gt, os.path.join(LOG_DIR,'vis/epoch_%d_%.2f/gt_%d.ply'%(epoch, mean_chamfer_loss*4.883, i)))
        io_util.write_ply(pred, os.path.join(LOG_DIR,'vis/epoch_%d_%.2f/pred_%d.ply'%(epoch, mean_chamfer_loss*4.883, i)))
        io_util.write_ply(res, os.path.join(LOG_DIR,'vis/epoch_%d_%.2f/res_%d.ply'%(epoch, mean_chamfer_loss*4.883, i)))
        io_util.write_ply(Y2X, os.path.join(LOG_DIR,'vis/epoch_%d_%.2f/pred_Y2X_%d.ply'%(epoch, mean_chamfer_loss*4.883, i)))
        io_util.write_ply(Y2X2Y, os.path.join(LOG_DIR,'vis/epoch_%d_%.2f/pred_Y2X2Y_%d.ply'%(epoch, mean_chamfer_loss*4.883, i)))
        io_util.write_ply(X2Y2X, os.path.join(LOG_DIR,'vis/epoch_%d_%.2f/pred_X2Y2X_%d.ply'%(epoch, mean_chamfer_loss*4.883, i)))

    return mean_chamfer_loss*4.883


if __name__ == "__main__":
    np.random.seed(int(time.time()))
    tf.set_random_seed(int(time.time()))
    train()
    LOG_FOUT.close()
    LOG_RESULT_FOUT.close()
