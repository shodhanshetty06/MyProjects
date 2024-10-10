Requirements :

Python: python8 or python 9
Tensorflow  :  2.10.1
tensorboard :  2.10.1
protobuf    :  3.19.6
pandas      :  2.2.3
object-detection  :  0.1
keras       :  2.10.0
NVDIA CUDA  :  11.2
CuDNN       :  8.1.0
Protoc      :  24.9.1

(Article Link :https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html)

comand to be executed in (Object Detection Using Tensorflow\research )

conda create -n tensorflow pip python=3.9

conda activate tensorflow

protoc object_detection/protos/*.proto --python_out=.


#COCO API installation
pip install cython
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI


Install the Object Detection API
{(Git Bash:# From within TensorFlow/models/research/) 
cp object_detection/packages/tf2/setup.py .)}
python -m pip install .


#Test your Installation
python object_detection/builders/model_builder_tf2_test.py

