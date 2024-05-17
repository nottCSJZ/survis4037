define({ entries : {

    "Badrinarayanan2017SegNet": {
        "abstract": "We present a novel and practical deep fully convolutional neural network architecture for semantic pixel-wise segmentation termed SegNet. This core trainable segmentation engine consists of an encoder network, a corresponding decoder network followed by a pixel-wise classification layer. The architecture of the encoder network is topologically identical to the 13 convolutional layers in the VGG16 network [1] . The role of the decoder network is to map the low resolution encoder feature maps to full input resolution feature maps for pixel-wise classification. The novelty of SegNet lies is in the manner in which the decoder upsamples its lower resolution input feature map(s). Specifically, the decoder uses pooling indices computed in the max-pooling step of the corresponding encoder to perform non-linear upsampling. This eliminates the need for learning to upsample. The upsampled maps are sparse and are then convolved with trainable filters to produce dense feature maps. We compare our proposed architecture with the widely adopted FCN [2] and also with the well known DeepLab-LargeFOV [3] , DeconvNet [4] architectures. This comparison reveals the memory versus accuracy trade-off involved in achieving good segmentation performance. SegNet was primarily motivated by scene understanding applications. Hence, it is designed to be efficient both in terms of memory and computational time during inference. It is also significantly smaller in the number of trainable parameters than other competing architectures and can be trained end-to-end using stochastic gradient descent. We also performed a controlled benchmark of SegNet and other architectures on both road scenes and SUN RGB-D indoor scene segmentation tasks. These quantitative assessments show that SegNet provides good performance with competitive inference time and most efficient inference memory-wise as compared to other architectures. We also provide a Caffe implementation of SegNet and a web demo at http://mi.eng.cam.ac.uk/projects/segnet/.",
        "author": "Badrinarayanan, Vijay and Kendall, Alex and Cipolla, Roberto",
        "doi": "10.1109/TPAMI.2016.2644615",
        "issn": "1939-3539",
        "journal": "IEEE Transactions on Pattern Analysis and Machine Intelligence"
        "keywords": "Decoding;Neural networks;Training;Computer architecture;Image segmentation;Semantics;Convolutional codes;Deep convolutional neural networks;semantic pixel-wise segmentation;indoor scenes;road scenes;encoder;decoder;pooling;upsampling",
        "number": "12"
        "pages": "2481--2495",
        "publisher": "IEEE Computer Society",
        "series": "TPAMI",
        "title": "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation",
        "type": "article",
        "volume": "39",
        "year": "2017"
    },

    "Chen2016attention":{
        "abstract": "Incorporating multi-scale features in fully convolutional neural networks (FCNs) has been a key element to achieving state-of-the-art performance on semantic image segmentation. One common way to extract multi-scale features is to feed multiple resized input images to a shared deep network and then merge the resulting features for pixelwise classification. In this work, we propose an attention mechanism that learns to softly weight the multi-scale features at each pixel location. We adapt a state-of-the-art semantic image segmentation model, which we jointly train with multi-scale input images and the attention model. The proposed attention model not only outperforms averageand max-pooling, but allows us to diagnostically visualize the importance of features at different positions and scales. Moreover, we show that adding extra supervision to the output at each scale is essential to achieving excellent performance when merging multi-scale features. We demonstrate the effectiveness of our model with extensive experiments on three challenging datasets, including PASCAL-Person-Part, PASCAL VOC 2012 and a subset of MS-COCO 2014.",
        "address": "Las Vegas, NV, USA",
        "author": "Chen, Liang-Chieh and Yang, Yi and Wang, Jiang and Xu, Wei and Yuille, Alan L.",
        "doi": "10.1109/CVPR.2016.396",
        "issn": "1063-6919",
        "conference": "2016 IEEE Conference on Computer Vision and Pattern Recognition",
        "keywords": "Adaptation models;Semantics;Image segmentation;Computational modeling;Training;Feature extraction;Computer vision",
        "pages": "3640--3649",
        "publisher": "IEEE Computer Society",
        "series": "CVPR",
        "title": "Attention to Scale: Scale-Aware Semantic Image Segmentation",
        "type": "inproceedings",
        "year": "2016"
    },

    "Chen2018asc":{
        "abstract": "Spatial pyramid pooling module or encode-decoder structure are used in deep neural networks for semantic segmentation task. The former networks are able to encode multi-scale contextual information by probing the incoming features with filters or pooling operations at multiple rates and multiple effective fields-of-view, while the latter networks can capture sharper object boundaries by gradually recovering the spatial information. In this work, we propose to combine the advantages from both methods. Specifically, our proposed model, DeepLabv3+, extends DeepLabv3 by adding a simple yet effective decoder module to refine the segmentation results especially along object boundaries. We further explore the Xception model and apply the depthwise separable convolution to both Atrous Spatial Pyramid Pooling and decoder modules, resulting in a faster and stronger encoder-decoder network. We demonstrate the effectiveness of the proposed model on PASCAL VOC 2012 and Cityscapes datasets, achieving the test set performance of 89{\%} and 82.1{\%} without any post-processing. Our paper is accompanied with a publicly available reference implementation of the proposed models in Tensorflow at https://github.com/tensorflow/models/tree/master/research/deeplab.",
        "address": "Munich, Germany",
        "author": "Chen, Liang-Chieh and Zhu, Yukun and Papandreou, George and Schroff, Florian and Adam, Hartwig",
        "doi": "10.1007/978-3-030-01234-2_49",
        "isbn": "978-3-030-01234-2",
        "conference": "Computer Vision – ECCV 2018",
        "keywords": "Segmantic image segmentation;Spatial pyramid pooling;Encoder-decoder;Depthwise separable convolution",
        "pages": "833--851",
        "publisher": "Springer International Publishing",
        "series": "ECCV",
        "title": "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation",
        "type": "inproceedings",
        "year": "2018"
    },

    "Chen2018maskLab":{
        "abstract": "In this work, we tackle the problem of instance segmentation, the task of simultaneously solving object detection and semantic segmentation. Towards this goal, we present a model, called MaskLab, which produces three outputs: box detection, semantic segmentation, and direction prediction. Building on top of the Faster-RCNN object detector, the predicted boxes provide accurate localization of object instances. Within each region of interest, MaskLab performs foreground/background segmentation by combining semantic and direction prediction. Semantic segmentation assists the model in distinguishing between objects of different semantic classes including background, while the direction prediction, estimating each pixel's direction towards its corresponding center, allows separating instances of the same semantic class. Moreover, we explore the effect of incorporating recent successful methods from both segmentation and detection (e.g., atrous convolution and hypercolumn). Our proposed model is evaluated on the COCO instance segmentation benchmark and shows comparable performance with other state-of-art models.",
        "address": "Salt Lake City, UT, USA",
        "author": "Chen, Liang-Chieh and Hermans, Alexander and Papandreou, George and Schroff, Florian and Wang, Peng and Adam, Hartwig",
        "doi": "10.1109/CVPR.2018.00422",
        "issn": "2575-7075",
        "conference": "2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition",
        "keywords": "Semantics;Agriculture;Image segmentation;Object detection;Convolution;Feature extraction;Encoding",
        "pages": "4013--4022",
        "publisher": "IEEE Computer Society",
        "series": "CVPR",
        "title": "MaskLab: Instance Segmentation by Refining Object Detection with Semantic and Direction Features",
        "type": "inproceedings",
        "year": "2018"
    },

    "Fu2019attention":{
        "abstract": "In this paper, we address the scene segmentation task by capturing rich contextual dependencies based on the self-attention mechanism. Unlike previous works that capture contexts by multi-scale features fusion, we propose a Dual Attention Networks (DANet) to adaptively integrate local features with their global dependencies. Specifically, we append two types of attention modules on top of traditional dilated FCN, which model the semantic interdependencies in spatial and channel dimensions respectively. The position attention module selectively aggregates the features at each position by a weighted sum of the features at all positions. Similar features would be related to each other regardless of their distances. Meanwhile, the channel attention module selectively emphasizes interdependent channel maps by integrating associated features among all channel maps. We sum the outputs of the two attention modules to further improve feature representation which contributes to more precise segmentation results. We achieve new state-of-the-art segmentation performance on three challenging scene segmentation datasets, i.e., Cityscapes, PASCAL Context and COCO Stuff dataset. In particular, a Mean IoU score of 81.5% on Cityscapes test set is achieved without using coarse data.",
        "address": "Long Beach, CA, USA",
        "author": "Fu, Jun and Liu, Jing and Tian, Haijie and Li, Yong and Bao, Yongjun and Fang, Zhiwei and Lu, Hanqing",
        "doi": "10.1109/CVPR.2019.00326",
        "issn": "2575-7075",
        "conference": "2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition",
        "keywords": "Segmentation;Grouping and Shape;Scene Analysis and Understanding",
        "pages": "3141--3149",
        "publisher": "IEEE Computer Society",
        "series": "CVPR",
        "title": "Dual Attention Network for Scene Segmentation",
        "type": "inproceedings",
        "year": "2019"
    },

    "Girshick2014feature":{
        "abstract": "Object detection performance, as measured on the canonical PASCAL VOC dataset, has plateaued in the last few years. The best-performing methods are complex ensemble systems that typically combine multiple low-level image features with high-level context. In this paper, we propose a simple and scalable detection algorithm that improves mean average precision (mAP) by more than 30% relative to the previous best result on VOC 2012 -- achieving a mAP of 53.3%. Our approach combines two key insights: (1) one can apply high-capacity convolutional neural networks (CNNs) to bottom-up region proposals in order to localize and segment objects and (2) when labeled training data is scarce, supervised pre-training for an auxiliary task, followed by domain-specific fine-tuning, yields a significant performance boost. Since we combine region proposals with CNNs, we call our method R-CNN: Regions with CNN features. We also present experiments that provide insight into what the network learns, revealing a rich hierarchy of image features. Source code for the complete system is available at http://www.cs.berkeley.edu/~rbg/rcnn.",
        "address": "Columbus, OH, USA",
        "author": "Girshick, Ross and Donahue, Jeff and Darrell, Trevor and Malik, Jitendra",
        "doi": "10.1109/CVPR.2014.81",
        "issn": "1063-6919",
        "conference": "2014 IEEE Conference on Computer Vision and Pattern Recognition",
        "keywords": "Proposals;Feature extraction;Training;Visualization;Object detection;Vectors;Support vector machines",
        "pages": "580--587",
        "publisher": "IEEE Computer Society",
        "series": "CVPR",
        "title": "Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation",
        "type": "inproceedings",
        "year": "2014"
    },
    
    "KH2017mrcnn":{
        "abstract": "Object detection performance, as measured on the canonical PASCAL VOC dataset, has plateaued in the last few years. The best-performing methods are complex ensemble systems that typically combine multiple low-level image features with high-level context. In this paper, we propose a simple and scalable detection algorithm that improves mean average precision (mAP) by more than 30% relative to the previous best result on VOC 2012 -- achieving a mAP of 53.3%. Our approach combines two key insights: (1) one can apply high-capacity convolutional neural networks (CNNs) to bottom-up region proposals in order to localize and segment objects and (2) when labeled training data is scarce, supervised pre-training for an auxiliary task, followed by domain-specific fine-tuning, yields a significant performance boost. Since we combine region proposals with CNNs, we call our method R-CNN: Regions with CNN features. We also present experiments that provide insight into what the network learns, revealing a rich hierarchy of image features. Source code for the complete system is available at http://www.cs.berkeley.edu/~rbg/rcnn.",
        "address": "Venice, Italy",
        "author": "He, Kaiming and Gkioxari, Georgia and DollÃ¡r, Piotr and Girshick, Ross",
        "doi": "10.1109/ICCV.2017.322",
        "issn": "2380-7504",
        "conference": "2017 IEEE International Conference on Computer Vision",
        "keywords": "Feature extraction;Image segmentation;Object detection;Semantics;Quantization (signal);Robustness",
        "pages": "2980--2988",
        "publisher": "IEEE Computer Society",
        "series": "ICCV",
        "title": "Mask R-CNN",
        "type": "inproceedings",
        "year": "2017"
    },

    "Lin2017refineNet":{
        "abstract": "Recently, very deep convolutional neural networks (CNNs) have shown outstanding performance in object recognition and have also been the first choice for dense classification problems such as semantic segmentation. However, repeated subsampling operations like pooling or convolution striding in deep CNNs lead to a significant decrease in the initial image resolution. Here, we present RefineNet, a generic multi-path refinement network that explicitly exploits all the information available along the down-sampling process to enable high-resolution prediction using long-range residual connections. In this way, the deeper layers that capture high-level semantic features can be directly refined using fine-grained features from earlier convolutions. The individual components of RefineNet employ residual connections following the identity mapping mindset, which allows for effective end-to-end training. Further, we introduce chained residual pooling, which captures rich background context in an efficient manner. We carry out comprehensive experiments and set new state-of-the-art results on seven public datasets. In particular, we achieve an intersection-over-union score of 83.4 on the challenging PASCAL VOC 2012 dataset, which is the best reported result to date.",
        "address": "Honolulu, HI, USA",
        "author": "Lin, Guosheng and Milan, Anton and Shen, Chunhua and Reid, Ian",
        "doi": "10.1109/CVPR.2017.549",
        "issn": "1063-6919",
        "conference": "2017 IEEE Conference on Computer Vision and Pattern Recognition",
        "keywords": "Semantics;Convolution;Image segmentation;Image resolution;Training;Computer architecture",
        "pages": "5168--5177",
        "publisher": "IEEE Computer Society",
        "series": "CVPR",
        "title": "RefineNet: Multi-path Refinement Networks for High-Resolution Semantic Segmentation",
        "type": "inproceedings",
        "year": "2017"
    },

    "Long2015fcn":{
        "abstract": "Convolutional networks are powerful visual models that yield hierarchies of features. We show that convolutional networks by themselves, trained end-to-end, pixels-to-pixels, exceed the state-of-the-art in semantic segmentation. Our key insight is to build {\^a}€{\oe}fully convolutional{\^a}€ networks that take input of arbitrary size and produce correspondingly-sized output with efficient inference and learning. We define and detail the space of fully convolutional networks, explain their application to spatially dense prediction tasks, and draw connections to prior models. We adapt contemporary classification networks (AlexNet [20], the VGG net [31], and GoogLeNet [32]) into fully convolutional networks and transfer their learned representations by fine-tuning [3] to the segmentation task. We then define a skip architecture that combines semantic information from a deep, coarse layer with appearance information from a shallow, fine layer to produce accurate and detailed segmentations. Our fully convolutional network achieves state-of-the-art segmentation of PASCAL VOC (20% relative improvement to 62.2% mean IU on 2012), NYUDv2, and SIFT Flow, while inference takes less than one fifth of a second for a typical image.",
        "address": "Boston, MA, USA",
        "author": "Long, Jonathan and Shelhamer, Evan and Darrell, Trevor",
        "doi": "10.1109/CVPR.2015.7298965",
        "issn": "1063-6919",
        "conference": "2015 IEEE Conference on Computer Vision and Pattern Recognition",
        "keywords": "Semantics;Training;Convolution;Image segmentation;Computer architecture;Deconvolution;Adaptation models",
        "pages": "3431-3440",
        "publisher": "IEEE Computer Society",
        "series": "CVPR",
        "title": "Fully convolutional networks for semantic segmentation",
        "type": "inproceedings",
        "year": "2015"
    },

    "Ronneberger2015UNet":{
        "abstract": "There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. We show that such a network can be trained end-to-end from very few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. Using the same network trained on transmitted light microscopy images (phase contrast and DIC) we won the ISBI cell tracking challenge 2015 in these categories by a large margin. Moreover, the network is fast. Segmentation of a 512x512 image takes less than a second on a recent GPU. The full implementation (based on Caffe) and the trained networks are available at http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net.",
        "address": "Munich, Germany",
        "author": "Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas",
        "doi": "10.1007/978-3-319-24574-4_28",
        "isbn": "978-3-319-24574-4",
        "conference": "Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2015",
        "keywords": "Training image;Data augmentation;Convolution layer;Deep network;Ground truth segmentation",
        "pages": "234--241",
        "publisher": "Springer International Publishing",
        "series": "MICCAI",
        "title": "U-Net: Convolutional Networks for Biomedical Image Segmentation",
        "type": "inproceedings",
        "year": "2015"
    }

}});