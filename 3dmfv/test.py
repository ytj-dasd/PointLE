import os
import sys
import numpy as np

import matplotlib
matplotlib.use('pdf')
# import matplotlib.pyplot as plt
import importlib
import argparse
import tensorflow as tf
import pickle
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tf_util
import visualization
import provider
import utils

model_num = 1

# ModelNet40 official train/test split. MOdelNet10 requires separate downloading and sampling.
MAX_N_POINTS = 2048
NUM_CLASSES = 41
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet'+str(40)+'_ply_hdf5_'+ str(MAX_N_POINTS)+ '/train_files1.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet'+str(40)+'_ply_hdf5_'+ str(MAX_N_POINTS)+ '/vehicle_test_files_testcar.txt'))



LABEL_MAP = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet'+str(40)+'_ply_hdf5_'+ str(MAX_N_POINTS)+ '/shape_names.txt'))

print( "Loading Modelnet" + str(NUM_CLASSES))

#Execute
#python train_cls.py  --gpu=0 --log_dir='log' --batch_size=64 --num_point=1024 --num_gaussians=8 --gmm_variance=0.0156 --gmm_type='grid' --learning_rate=0.001  --model='voxnet_pfv' --max_epoch=200 --momentum=0.9 --optimizer='adam' --decay_step=200000  --weight_decay=0.0 --decay_rate=0.7

augment_rotation, augment_scale, augment_translation, augment_jitter, augment_outlier = (False, True, True, True, False)

parser = argparse.ArgumentParser()
#Parameters for learning
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='3dmfv_net_cls', help='Model name [default: 3dmfv_net_cls]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 64]')
parser.add_argument('--model_path', default='log/modelnet41/3dmfv_net_cls/grid5_log_trial/13model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--log_dir', default='log_trial', help='Log dir [default: log]')
parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores from multiple rotations [default: 1]')

# Parameters for GMM
parser.add_argument('--gmm_type',  default='grid', help='type of gmm [grid/learn], learn uses expectation maximization algorithm (EM) [default: grid]')
parser.add_argument('--num_gaussians', type=int , default=5, help='number of gaussians for gmm, if grid specify subdivisions, if learned specify actual number[default: 5, for grid it means 125 gaussians]')
parser.add_argument('--gmm_variance', type=float,  default=0.04, help='variance for grid gmm, relevant only for grid type')
FLAGS = parser.parse_args()


N_GAUSSIANS = FLAGS.num_gaussians
GMM_TYPE = FLAGS.gmm_type
GMM_VARIANCE = FLAGS.gmm_variance

BATCH_SIZE = FLAGS.batch_size
#NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu
MODEL_PATH = FLAGS.model_path
num_votes=FLAGS.num_votes

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = 'log/modelnet' + str(NUM_CLASSES) + '/' + FLAGS.model + '/'+ GMM_TYPE + str(N_GAUSSIANS) + '_' + FLAGS.log_dir



def test(gmm,NUM_POINT):
    is_training = False
    with tf.device('/gpu:'+str(GPU_INDEX)):
        points_pl, labels_pl, w_pl, mu_pl, sigma_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, gmm )
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # Get model and loss 
            # pred, end_points = MODEL.get_model(points_pl, is_training_pl)
            # MODEL.get_loss(pred, labels_pl, end_points)
            # losses = tf.get_collection('losses')
            # total_loss = tf.add_n(losses, name='total_loss')
        pred, fv = MODEL.get_model(points_pl, w_pl, mu_pl, sigma_pl, is_training_pl,  add_noise=False, num_classes=NUM_CLASSES)
        loss = MODEL.get_loss(pred, labels_pl)
        tf.summary.scalar('loss', loss)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()    

    # Create a session
    sess = tf_util.get_session(GPU_INDEX, limit_gpu=True)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    print("Model restored.")    
     
    ops = {'points_pl': points_pl,
               'labels_pl': labels_pl,
               'w_pl': w_pl,
               'mu_pl': mu_pl,
               'sigma_pl': sigma_pl,
               'is_training_pl': is_training_pl,
               'fv': fv,
               'pred': pred,
               'loss': loss}
    return sess,ops    


def test_one_epoch(gmm):
    is_training = False
    
    fout = open(os.path.join('pred_label.txt'), 'a')

    for fn in range(len(TEST_FILES)):
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn], compensate=False)
        NUM_POINT = current_data.shape[1]
        file_name = TEST_FILES[fn].split('_')[3]
        print("pointcloud size: ", NUM_POINT)
        points_idx = range(NUM_POINT)
        current_data = current_data[:, points_idx, :]
        #current_label = np.squeeze(current_label)

        file_size = current_data.shape[0]
        num_batches = file_size / BATCH_SIZE
        sess,ops = test(gmm,NUM_POINT)

        for batch_idx in range(num_batches):
           
            batch_pred_sum = np.zeros((BATCH_SIZE, NUM_CLASSES)) # score for classes
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE
            #t1 = time.time()

            for vote_idx in range(num_votes):
            # Shuffle point order to achieve different farthest samplings
                shuffled_indices = np.arange(NUM_POINT)
                np.random.shuffle(shuffled_indices)
                rotated_data = provider.rotate_point_cloud_by_angle(current_data[start_idx:end_idx, :, :] ,
                    vote_idx/float(num_votes) * np.pi * 2)                
                feed_dict = {ops['points_pl']: rotated_data ,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['w_pl']: gmm.weights_,
                         ops['mu_pl']: gmm.means_,
                         ops['sigma_pl']: np.sqrt(gmm.covariances_),
                         ops['is_training_pl']: is_training}
                loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
                batch_pred_sum += pred_val
           
                object_score = sess.run(tf.nn.softmax(batch_pred_sum))   
                #t2 = time.time()
                sort_id = np.argsort(object_score[0])
                fout.write(file_name)
                for i in range(-1,-5,-1):
                    fout.write(' %d %f ' % (sort_id[i],object_score[0,sort_id[i]]))
                fout.write(' %d %f\n' % (sort_id[-5],object_score[0,sort_id[-5]]))
        tf.reset_default_graph()

if __name__ == "__main__":

    gmm = utils.get_3d_grid_gmm(subdivisions=[N_GAUSSIANS, N_GAUSSIANS, N_GAUSSIANS], variance=GMM_VARIANCE)
    pickle.dump(gmm, open(os.path.join(LOG_DIR, 'gmm.p'), "wb"))
    test_one_epoch(gmm)
    #export_visualizations(gmm, LOG_DIR,n_model_limit=None)


