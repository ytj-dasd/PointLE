# TF1.2
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /home/dandelion/anaconda3/envs/pointnet++/lib/python2.7/site-packages/tensorflow/include -I /usr/local/cuda-11.2/include -lcudart -L /usr/local/cuda-11.2/lib64/ -O2 

# TF1.4
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /home/ytj/anaconda3/envs/pointnet2/lib/python2.7/site-packages/tensorflow/include -I /usr/local/cuda-10.1/include -I /home/ytj/anaconda3/envs/pointnet2/lib/python2.7/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-10.1/lib64/ -L/home/ytj/anaconda3/envs/pointnet2/lib/python2.7/site-packages/tensorflow -l:libtensorflow_framework.so.1 -O2 #-D_GLIBCXX_USE_CXX11_ABI=0
