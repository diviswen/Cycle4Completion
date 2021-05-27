nvcc = /usr/local/cuda-8.0/bin/nvcc
#nvcc = /usr/local/cuda/bin/nvcc
cudalib =  /usr/local/cuda-8.0/lib64
#tensorflow = /orions4-zfs/projects/optas/Virt_Env/tf_1.3/lib/python2.7/site-packages/tensorflow/include
tensorflow = /usr/local/lib/python2.7/dist-packages/tensorflow/include
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')


all: tf_approxmatch_so.so tf_approxmatch_g.cu.o tf_nndistance_so.so tf_nndistance_g.cu.o


tf_approxmatch_so.so: tf_approxmatch_g.cu.o tf_approxmatch.cpp
	g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -I $TF_INC -I $IF_INC/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L $TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0


tf_approxmatch_g.cu.o: tf_approxmatch_g.cu
	$(nvcc)  -std=c++11 -c -o tf_approxmatch_g.cu.o tf_approxmatch_g.cu -I $TF_INC -I $TF_INC/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CXXFLAGS


tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
	g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -I $TF_INC -I $IF_INC/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L $TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0


tf_nndistance_g.cu.o: tf_nndistance_g.cu
	$(nvcc)  -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $TF_INC -I$TF_INC/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CXXFLAGS


clean:
	rm tf_approxmatch_so.so
	rm tf_nndistance_so.so
	rm  *.cu.o 
