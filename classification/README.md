# Transfer Learning for Image Classification

In this tutorial, you will learn classify images by using transfer learning from a pre-trained Convolutional Neural Network (CNN).

A pre-trained model is a saved network that was previously trained on a large dataset, typically on a large-scale image classification task. For example, ImageNet has more than 14 million images while COCO dataset has more than 300K images.

In practice, very few people train an entire CNN from scratch (with random initialization), because it is relatively rare to have a dataset of sufficient size. You either use the pre-trained model as is (if the model already support your task) or use transfer learning to customize this model to a given task.

The intuition behind transfer learning for image classification is that if a model is trained on a large and general enough dataset, this model will effectively serve as a **generic model of the visual world**. You can then take advantage of these learned feature maps without having to start from scratch by training a large model on a large dataset.

![Alt text](md_images/cnn_featuremaps.png?raw=true "cnn_featuremaps")<br/>
[Source: S. Vignesh](https://medium.com/analytics-vidhya/the-world-through-the-eyes-of-cnn-5a52c034dbeb)

Two major transfer learning scenarios look as follows:

1. **Finetuning**: Instead of random initialization, we initialize the network with a pre-trained network. Depending on dataset size, you may unfreeze all layers or a few of the top layers of a pre-trained model base and jointly train both the newly-added classifier layers and the top layers of the base model. This allows us to "fine-tune" the higher-order feature representations in the base model in order to make them more relevant for the specific task.

2. **CNN as fixed feature extractor**: Here, we will freeze the weights for all of the network except that of the final fully connected (fc) layer. This last fc layer is replaced with a new one with random weights and only this layer is trained. You do not need to (re)train the entire model. In other words, you simply add a new classifier (either just re-train the last fc layer or just use any classifiers like SVM, Random Forest, etc), on top of the pretrained model so that you can repurpose the feature maps learned previously for the dataset. Classical example: OverFeat [Sermanet et. al 2013] used first 5 layers of CNN as feature extractor.

Once upon a time, in Year 2013 I was working on backpack classification task:
![Alt text](md_images/backpack_classification.jpg?raw=true "backpack_classifcation")


## When to use transfer learning?
CNN features are more generic in early layers and more original-dataset-specific in later layers. Here are some common rules of thumb:

![Alt text](md_images/when_to_use_TL.jpg?raw=true "whentouseTL")<br/>
[Source: Stanford CS231n](https://cs231n.github.io/)<br/>

Similar dataset - ImageNet-like natural scene images<br/>
Different dataset - microscope images, X-ray images
