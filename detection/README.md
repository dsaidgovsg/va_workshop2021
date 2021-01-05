# Object Detection

Object detection is the task of detecting instances of objects of a certain class within an image. The state-of-the-art deep learning methods can be categorized into two main types: one-stage methods and two stage-methods. One-stage methods prioritize inference speed, and example models include YOLO, SSD and RetinaNet. Two-stage methods prioritize detection accuracy, and example models include Faster R-CNN, Mask R-CNN and Cascade R-CNN.

## Brief History Before Deep Learning
* Early object detectors were based on **handcrafted** features
* Sliding window classifier, check if feature response is strong enough, if so: output detection.
* Classical examples:
    * Haar-like features [Viola and Jones (2001)]
        * Handcrafted weak features, calculate in sliding window using integral image, use boosted classifier like AdaBoost. Name comes from similarity to Haar wavelets.
        * Still supported in OpenCV
        ![Alt text](md_images/haar.jpg?raw=true "HaarDetector")

    * Histogram of Oriented Gradients [Dalal and Triggs(2005)]
        * Computes histogram of gradient orientation (HOG features) over sub-image blocks, extracted features are trained using Support Vector Machine (SVM)
        * Still supported in OpenCV
        ![Alt text](md_images/hog.jpg?raw=true "HogDetector")
    * Deformable Part Models [Felzenszwalb et al.(2008)]
        * Learn the relationship between HOG features of object parts via latent SVM
        ![Alt text](md_images/dpm.jpg?raw=true "DpmDetector")



## Deep Learning based Methods
* Two-stage methods
    * Operates in two serial stages:
        * Generate region proposal (instead of sliding window).
        * Classify each proposed region, if feature response strong enough, output detection
    * Prioritize detection accuracy but computationally heavy
    * Training is very slow as it happens in multiple phases (e.g. training region proposal & classifier iteratively)
        ![Alt text](md_images/twostage.jpg?raw=true "twostagemethods")
    * Examples: R-CNN, Fast R-CNN, Faster R-CNN
* One-stage methods (aka Single-shot)
    * The tasks of object localization and classification are fone in a single forward pass of the network
    * Prioritize inference speed
    ![Alt text](md_images/onestage.jpg?raw=true "onestagemethods")
    * Examples: Single Shot Detector (SSD), You Only Look Once (YOLO)


# Evaluation Metrics
For more detailed explanations please refer to:
1. https://blog.zenggyu.com/en/post/2018-12-16/an-introduction-to-evaluation-metrics-for-object-detection/
2. https://github.com/rafaelpadilla/Object-Detection-Metrics

## Basic Concepts
### Intersection Over Union (IOU)
IOU measures the overlap between two bounding boxes. IOU between a ground truth bounding box (in green) and a detected bounding box (in red).<br/>
![Alt text](md_images/iou.png?raw=true "iou")

### True Positive (TP), False Positive(FP), False Negative (FN)  
- *True Positive (TP)*: A correct detection. Detection with IOU ≥ threshold
- *False Positive (FP)*: A wrong detection. Detection with IOU < threshold
- *False Negative (FN)*: A ground truth not detected
- *True Negative (TN)*: Does not apply. It would represent a corrected misdetection. In the object detection task there are many possible bounding boxes that should not be detected within an image. Thus, TN would be all possible bounding boxes that were correctly not detected (so many possible boxes within an image). That's why it is not used by the metrics.

threshold: depending on the metric, it is usually set to 50%, 75% or 95%.

### Precision
Precision is the ability of a model to identify only the relevant objects. It is the percentage of correct positive predictions and is given by:
![Alt text](md_images/Precision.jpg?raw=true "precision")

### Recall
Recall is the ability of a model to find all the relevant cases (all ground truth bounding boxes). It is the percentage of true positive detected among all relevant ground truths and is given by:<br/>
![Alt text](md_images/Recall.jpg?raw=true "recall")

### Precision-Recall Curve
By setting the threshold for confidence score at different levels, we get different pairs of precision and recall:
![Alt text](md_images/PR_curve.gif?raw=true "pr_curve")<br/>
Note that as the threshold for confidence score decreases, recall increases monotonically; precision can go up and down (goes down with FPs and goes up again with TPs.), but the general tendency is to decrease.

### Average Precision (AP)
Comparing different curves (different detectors) is not an easy task. Therefore, average precision is used to calculate the area under the PR-curve. In essense, AO us the precision averaged across all unique recall levels. The curve can be interpolated using 11-point interpolation: precision is averaged at a set of eleven equally spaced recall levels [0, 0.1, 0.2, ..., 1] or at all unique recall levels presented by the data as the follows:<br/>
![Alt text](md_images/AP_interpolation.gif?raw=true "ap_interpolation")

### Mean Average Precision (mAP)
The calculation of AP only involves one class. However, in object detection there are usually more than one class (K>1). mAP is defined as the mean of all AP across all K classes. Note that mAP also depends on the IOU threshold used. Higher IOU threshold (stricter) will result in lower mAP. Typically, mAP reported will be at certain IOU threshold (e.g. mAP@IoU=0.5, mAP@IoU=0.75).<br/>
![Alt text](md_images/PR-curve-at-diff-IOUs.png?raw=true "pr_diff_iou")


# Custom Object Detector using YOLOv3 (Darknet Implementation)
For this demo, we will be using popular YOLOv3 Darknet implementation from https://github.com/AlexeyAB/darknet. The setup and training will be demonstrated using Google Colab notebook. However, if you have access to local GPU resource it will be easier to setup locally. This is because Darknet is implemented in C and CUDA, it requires environment setup and C compilation. Each time Colab session disconnected, the setup process has to be repeated.


## Data Preparation
In this tutorial, we will be using logo dataset which contains 100 images from two classes: McDonalds and Starbucks. You may download them from this [link](https://drive.google.com/file/d/112pZx-vRgh7TBVfnryPA-IJ2TGF8cFgY/view?usp=sharing).

YOLO Darknet requires a .txt file for each image with a line for each ground truth object in the image that looks like:
```
<object-class> <x> <y> <width> <height>
```
where:
- object-class: integer number of object from 0 to (N_Classes-1)
- x, y, width, height: float values relative to the image's width and height between 0 to 1. Note that x and y are center of rectangle (not top-left corner)

To annotate the images, we could use the following annotation tools that support YOLO format:
1. VideoIO - developed by in-house VA team, please request a copy from https://go.gov.sg/szh5nk<br/>
BONUS: VideoIO comes with built-in object classification and detection (YOLOv3) capabilities!<br/><br/>
![Alt text](md_images/videoio_demo.jpg?raw=true "videoiodemo")

2. labelimg - open source tool https://github.com/tzutalin/labelImg<br/><br/>
![Alt text](md_images/labelimg_demo.jpg?raw=true "labelimgdemo")

Once the images have been annotated, we can split the images into two sets using the **splitTrainAndTest.py** scripts:
1. Training set: this is the part of the data on which we train the model, typically we  randomly select between 70-90% of the data for training.
2. Test set: This is the part of the data on which we test the model, typically we set 10-30% of the data. No training image should be in test set.

To run the script, pass the path to image dataset as an argument:
```
python3 splitTrainAndTest.py /path/to/image_dataset
```
The script will produce two files: logo_train.txt and logo_test.txt. Move the generated text files into the image dataset folder.

## Config Files Preparation
YOLO Darknet needs three specific files to know how and what to train:
1. **logo.names** - this file contains the class names. Every new class should be on a new line
```
McDonalds
Starbucks
```
2. **logo.data** - this file defines the number of classes and path to image dataset folder. backup parameter defines the output folder that will hold training snapshots and final model weight. You need to create the folder manually if it doesn't exist.
```
classes = 2
train = /path/to/image_dataset/logo_train.txt
valid = /path/to/image_dataset/logo_test.txt
names = /path/to/image_dataset/logo.names
backup = backup/
```
3. **yolov3-tiny.cfg** - this is the config file that defines model architecture and training hyperparameters. In Darknet repo, there are many different models to choose from, in particular, there are four variants of yolov3 configurations:
* `yolov3-openimages.cfg` (247 MB COCO **Yolo v3**) - requires 4 GB GPU-RAM: https://pjreddie.com/media/files/yolov3-openimages.weights
* `yolov3-spp.cfg` (240 MB COCO **Yolo v3**) - requires 4 GB GPU-RAM: https://pjreddie.com/media/files/yolov3-spp.weights
* `yolov3.cfg` (236 MB COCO **Yolo v3**) - requires 4 GB GPU-RAM: https://pjreddie.com/media/files/yolov3.weights
* `yolov3-tiny.cfg` (34 MB COCO **Yolo v3 tiny**) - requires 1 GB GPU-RAM:  https://pjreddie.com/media/files/yolov3-tiny.weights
For demo purpose, this tutorial will use the most lightweight model (yolov3-tiny.cfg) for logo detection task. Download the weight and placed it in Darknet root folder:
```
wget https://pjreddie.com/media/files/yolov3-tiny.weights
```
Copy *cfg/yolov3-tiny.cfg* to the image dataset folder and rename it to *yolov3-tiny-logo-train.cfg* to differentiate from the original config file. Open the duplicated config file
and change the *filters* and *classes* values:

- Line 127: set filters=(classes + 5)*3 in our case filters=21
- Line 135: set classes=2, the number of categories we want to detect
- Line 171: set filters=(classes + 5)*3 in our case filters=21
- Line 177: set classes=2, the number of categories we want to detect

Some important training hyperparameters inside the config file:
```
[net]
#Testing
#batch=1
#subdivisions=1
#Training
batch=8
subdivisions=1
width=416
height=416
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1
learning_rate=0.001
#burn_in=1000
max_batches = 3000
policy=steps
steps=2000
scales=.1
```
- **batch**: defines the batch size during training. When the batch size is set to 8, it means 8 images are used in one iteration to update the weights of the neural network. Depending on the underlying hardware, you may experiment with higher batch size such as 16, 32, 64.
- **subdivisions**: even though you may want to use a batch size of 64, the GPU may not have enough memory to fit the batch data. Darknet allows you to specify subdivisions which process a fraction of the batch size (i.e., batch/subdivision) at one time on GPU before updating the weights. You can start the training with subdivisions=1, and if you get an Out of memory error, increase the subdivisions parameter by multiples of 2(e.g. 2, 4, 8, 16) till the training proceeds successfully. For this demo, we will leave subdivisions=1.
Note: During testing, both batch and subdivision are set to 1. Uncomment Line 3 and 4 and comment line 6 and 7 and save the config file as *yolov3-tiny-logo-test.cfg*
- **width, height, channels**: input training images are resized to width x height. For this demo we use the default values of 416x416. Larger size may improve result but more computationally expensive. Channels=3 indicates 3-channel RGB input.
- **momentum, decay**: regularization parameters for weight updating. Leave them as default.
- **learning_rate**: control how aggressively we should learn from current batch of data. Leave it as default 0.001
angle, saturation, exposure, hue: data augmentation to randomly transform training images
- **burn_in**: once this parameter is set, when iteration is less than burn_in, the learning rate is updated according to lr = base_lr * power(batch_num/burn_in,pwr) instead of based on policy. The assumption is that global advantage is near the initial position of the network after the start of training. therefore the learning rate changes from small to large. After the number of updates exceeds burn_in, the configured learning rate update strategy changes from large to small. Since this demo dataset is small, we will comment out this parameter.
- **max_batches**: maximum number of training iterations, set to 3000 for this demo.
- **policy, steps, scales**: control how learning rate is  decreased over iterations, e.g. learning rate starts at 0.001 and decreased to 0.001 x scales after iteration 2000.

# What's Next
1. Try out Tensorflow Object Detection API https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/
2. Try out latest YOLOv5 https://github.com/ultralytics/yolov5
3. Build video processing pipeline by adding object tracker such as DeepSORT https://github.com/nwojke/deep_sort
