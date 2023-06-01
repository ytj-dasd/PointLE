import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import modelnet_dataset
import modelnet_h5_dataset_test as modelnet_h5_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_cls_ssg', help='Model name. [default: pointnet2_cls_ssg]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 16]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log/11model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--normal', action='store_true', help='Whether to use normal information')
parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores from multiple rotations [default: 1]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
#NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_test.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 41
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(ROOT_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))] 

HOSTNAME = socket.gethostname()

    #TRAIN_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=True)
TEST_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/val.txt'), batch_size=BATCH_SIZE, npoints=1024,shuffle=False)
FILE_LEN = TEST_DATASET.file_len()
fout = open(os.path.join('output.txt'), 'a')
    

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def test(NUM_POINT):
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
        MODEL.get_loss(pred, labels_pl, end_points)
        losses = tf.get_collection('losses')
        total_loss = tf.add_n(losses, name='total_loss')
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    var=tf.global_variables()
    print('restore_variables', var)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': total_loss}
    return sess,ops

def test_one_epoch(num_votes):
    is_training = False
    #is_training = True
    for file_id in range(FILE_LEN):
        with tf.Graph().as_default():
            NUM_POINT = TEST_DATASET.point_num()
            print("pointcloud size: ", NUM_POINT)

            #with tf.Graph().as_default():   
            sess,ops = test(NUM_POINT)
            # Make sure batch data is of same size
            cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TEST_DATASET.num_channel()))
            cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)
            batch_idx = 0
            while TEST_DATASET.has_next_batch():
                batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
                bsize = batch_data.shape[0]
                print('Batch: %03d, batch size: %d'%(batch_idx, bsize))
                # for the last batch in the epoch, the bsize:end are from last batch
                cur_batch_data[0:bsize,...] = batch_data
                cur_batch_label[0:bsize] = batch_label
            
                #t1 = time.time()
                batch_pred_sum = np.zeros((BATCH_SIZE, NUM_CLASSES)) # score for classes
                for vote_idx in range(num_votes):
                    # Shuffle point order to achieve different farthest samplings
                    shuffled_indices = np.arange(NUM_POINT)
                    np.random.shuffle(shuffled_indices)
                    if FLAGS.normal:
                        rotated_data = provider.rotate_point_cloud_by_angle_with_normal(cur_batch_data[:, shuffled_indices, :],vote_idx/float(num_votes) * np.pi * 2)
                    else:
                        rotated_data = provider.rotate_point_cloud_by_angle(cur_batch_data[:, shuffled_indices, :],vote_idx/float(num_votes) * np.pi * 2)
                    feed_dict = {ops['pointclouds_pl']: rotated_data,
                            ops['labels_pl']: cur_batch_label,
                            ops['is_training_pl']: is_training}
                    loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
                    name = [n.name for n in tf.get_default_graph().as_graph_def().node]
                    #print(name)
                    output = sess.run(tf.get_default_graph().get_operation_by_name('fc2/bn/Reshape_1').outputs[0], feed_dict=feed_dict)
                    for k in range(255):
                        fout.write(str(output[0][k])+' ')
                    fout.write(str(output[0][255])+'\n')
                    #print(output)
                    batch_pred_sum += pred_val
                    
                object_score = sess.run(tf.nn.softmax(batch_pred_sum))   
                #print('score: ',object_score)
                #print('pred: ',batch_pred_sum)
                #t2 = time.time()
        
                # sort_id = np.argsort(object_score)
                # #print(sort_id)
                # for i in range(-1,-3,-1):
                #     fout.write('%d ' % (sort_id[0,i]))
                # fout.write('%d \n' % (sort_id[0,-3]))
                batch_idx += 1
        tf.reset_default_graph()
    



if __name__=='__main__':
    test_one_epoch(num_votes=FLAGS.num_votes)
    LOG_FOUT.close()
