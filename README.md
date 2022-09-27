# k_mnist_deep_neural_network
A  deep neural network to classify k-mnist data.

** Damith Senadeera **

**Introduction**

The main tasks of this work are to load the k-mnist data set and implement the stem, backbone and the classifier blocks and define the main neural network and experiment to classify the k-mnist dataset to gain the maximum testing accuracy.

**Task 1**

The task 1 is to load the k-mnist data set divided into the training and testing data sets. I used the PyTorch package's "torchvision.datasets.KMNIST" function to download the training and testing data sets of k-mnist and then the images are resized if required and output the data according to the required batch-size.

An important thing to note is that when loading data for my experiments, I resized all the k-mnist images (training and testing) into the resolution of 32 x 32 pixels form their original resolution of 28 x 28 pixels, in order to experiment with patch size parameter in power of 2 (such as 2, 4, 8, etc.)

**Task 2**

The task 2 is divided into 2 main sub parts namely, Stem block definition and Backbone block definition

**Part 1 – Stem Block Definition**

To define the stem block to extract non-overlapping patches of the image, I used the PyTorch function "torch.nn.Unfold" which outputs patches from images according to the given kernel size (which is the patch size) and the stride. Since in this case I need non-overlapping patches, I made sure the kernel size is always equal to the stride.

In addition to the basic structures expected in the Stem block, I included a batch normalization layer to make the network converge faster and learn faster along with a ReLU activation layer. The following block diagram depicts my Stem block implementation.

<p>&nbsp;</p>
<kbd>
<img src="https://user-images.githubusercontent.com/14356479/192408325-9594dd53-5b59-42d7-a3ef-2d0f5dedc1b5.jpg"  width="500" ></kbd>
<p>&nbsp;</p>


**Part 2 – Backbone Block Definition**

To define the backbone block, I take the no. of patches, user defined dimension used in the stem block and two hidden dimensions as inputs in the initialization and in addition to the basic 2 MLPs of the block, I include Batch Normalization layers and a skip connection from the beginning of the block to the end of the 2nd MLP taking inspiration from the ResNet Architecture to alleviate the problem of degradation in the network as I intend to stack backbone blocks over one another. Also, all the non-linear activation functions used in this block are ReLU functions in order to alleviate the vanishing gradient problem as I intend to use multiple layers in the network. The following block diagram depicts my Backbone block implementation.

<p>&nbsp;</p>
<kbd>
<img src="https://user-images.githubusercontent.com/14356479/192408369-0356ab96-2042-4ad5-a49c-d490b47927db.jpg"  width="750" ></kbd>
<p>&nbsp;</p>


**Task 3**

I used the PyTorch Cross Entropy loss as my loss function to update the weights of my networks and I used the Adam Optimizer for the initial experiments to get an idea of how the networks learn with quite a low learning rate for a small number of epochs and for the final model I used SGD Optimizer with the use of a learning rate scheduler to reduce the learning rate after training for a certain number of epochs.

**Task 4**

In this task, I created the training function to train the network and produce the training loss, training accuracy and testing accuracy curves by referring to the d2l training functions and experimented with various combinations of hyper-parameters to train the networks with the target of obtaining the best test accuracy.

In general for all the experimented networks, for the initialization of weights, I used the PyTorch default weight initialization (Kaiming He uniform initialization) method and as the batch size I used 128 image samples per batch.

Experiment 1, 2 and 3

For the 1st 3 experiments I used just one Backbone block.

The following table provides details on the 1st 3 experiments.

| **Experiment No.** | **Common Hyper-Parameters** | **Unique Hyper- Parameters** | **Final Training Loss** | **Final Training Accuracy** | **Final Testing Accuracy** |
| --- | --- | --- | --- | --- | --- |
| Exp - 1 | Image Resolution = 32x32, No. of Backbone blocks = 1, For Backbone block 1 - Hidden dimensions 1 and 2 = 64, Transform dimension for stem (no\_d) = 128, Batch Size = 128, Learning Rate = 0.001, No. of Epochs = 10, Weight Decay = 0.0001, Optimizer = Adam | Patch Size (ptch\_sz) = 2 | 0.21563 | 93.93% | 82.64% |
| Exp - 2 | Patch Size (ptch\_sz) = 4 | 0.20785 | 94.04% | **85.05%** |
| Exp - 3 | Patch Size (ptch\_sz) = 8 | 0.21177 | 93.88% | 84.86% |

Out of these 3 experiments the training loss, training accuracy and testing accuracy curves for the best 2 test accuracies of 85.05% and 84.86% which were reached in experiments 2 and 3 are shown below.


<p>&nbsp;</p>
<kbd>
<img src="https://user-images.githubusercontent.com/14356479/192408470-06df5097-c282-4e41-80cb-439e1cff3b2d.jpg"  width="500" ></kbd>
<p>&nbsp;</p>
Figure 1: Metric curves for Experiment 2 


<p>&nbsp;</p>
<kbd>
<img src="https://user-images.githubusercontent.com/14356479/192408501-a32a2065-7fdb-4eb8-897f-e73c7d0e7af3.jpg"  width="500" ></kbd>
<p>&nbsp;</p>
Figure 2: Metric curves for Experiment 3

Experiment 4 and 5

Since Patch size 4 and 8 form the above 3 experiments gave the best test accuracies, for the next 2 experiments, exp number 4 and 5 I used 3 Backbone blocks stacked and tested for patch sizes 4 and 8.

The following table provides details on those 2 experiments.

| **Experiment No.** | **Common Hyper-Parameters** | **Unique Hyper- Parameters** | **Final Training Loss** | **Final Training Accuracy** | **Final Testing Accuracy** |
| --- | --- | --- | --- | --- | --- |
| Exp - 4 | Image Resolution = 32x32, No. of Backbone blocks = 3, For Backbone block 1 - Hidden dimensions 1 and 2 = 64, For Backbone block 2 - Hidden dimensions 1 and 2 = 128, For Backbone block 3 - Hidden dimensions 1 and 2 = 256, Transform dimension for stem (no\_d) = 256, Batch Size = 128, Learning Rate = 0.001, No. of Epochs = 10, Weight Decay = 0.0001, Optimizer = Adam | Patch Size (ptch\_sz) = 4 | 0.06434 | 98.25% | **92.58%** |
| Exp - 5 | Patch Size (ptch\_sz) = 8 | 0.06118 | 98.30% | 90.74% |

The training loss, training accuracy and testing accuracy curves for the experiments 4 and 5 are shown below.

<p>&nbsp;</p>
<kbd>
<img src="https://user-images.githubusercontent.com/14356479/192408594-6a35c5cc-3bcd-4c5e-ad2b-28cbc96bafc7.jpg"  width="500" ></kbd>
<p>&nbsp;</p>
Figure 3: Metric curves for Experiment 4 

<p>&nbsp;</p>
<kbd>
<img src="https://user-images.githubusercontent.com/14356479/192408650-cad3706e-46c5-4b1d-a11f-77e8695ed199.jpg"  width="500" ></kbd>
<p>&nbsp;</p>
Figure 4: Metric curves for Experiment 5

**Task 5**

From the above experiments it was evident that when I add a greater number of Backbone blocks the testing accuracy of the model increases. Also, among the 5 experiments the highest test accuracies were obtained for a patch size of 4. Therefore, I defined the final model with 6 backbone blocks for a patch size of 4 and trained it with SGD optimizer with a learning rate scheduler to reduce the learning rate 30% after each 20 epochs and ran it for 50 epochs separately 3 times from separate initializations of the same model to see if the model was stable and test if the final test accuracies obtained converged to a similar percentage in all the runs.

| **Run No.** | **Hyper-Parameters of the Final Model** | **Final Training Loss** | **Final Training Accuracy** | **Final Testing Accuracy** |
| --- | --- | --- | --- | --- |
| Run - 1 | Image Resolution = 32x32, No. of Backbone blocks = 6, For Backbone blocks 1 and 2 - Hidden dimensions 1 and 2 = 64, For Backbone blocks 3 and 4 - Hidden dimensions 1 and 2 =128, For Backbone blocks 5 and 6 - Hidden dimensions 1 and 2 = 256, Patch Size (ptch\_sz) = 4, Transform dimension for stem (no\_d) = 256, Batch Size = 128 , Weight Decay = 0.0001, Optimizer = SGD, No. of Epochs = 50, Learning Rate for 1st 20 epochs = 0.05, Learning Rate for 2nd 20 epochs = 0.035, Learning Rate for final 10 epochs = 0.0245 | 0.00023 | 99.998% | 94.220% |
| Run - 2 | 0.00022 | 100.000% | 94.240% |
| Run - 3 | 0.00026 | 99.998% | 94.240% |
| **Average metrics over the 3 runs** | 0.00024 | 99.999% | **94.233%** |

The training loss, training accuracy and testing accuracy curves for the run no. 1 and no. 2 are shown below.

<p>&nbsp;</p>
<kbd>
<img src="https://user-images.githubusercontent.com/14356479/192408852-20b2ef67-5450-4879-95f2-560eea4ab983.jpg"  width="500" ></kbd>
<p>&nbsp;</p>
Figure 5: Metric curves for run no. 1 


<p>&nbsp;</p>
<kbd>
<img src="https://user-images.githubusercontent.com/14356479/192408885-ec70a908-7eb9-4323-82ef-375092bd6f47.jpg"  width="500" ></kbd>
<p>&nbsp;</p>
Figure 6: Metric curves for run no. 2
