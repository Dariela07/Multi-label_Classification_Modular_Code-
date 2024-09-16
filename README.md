# Multi-label Classification Project Overview

<div style="text-align: justify;">
This project develops an email classifier that categorizes emails into multiple classes simultaneously, using modular preprocessing for easy integration of new data or ML models. It employs TF-IDF embedding and includes preprocessing steps like deduplication and noise removal. 


A Chained Multi-Output methodology is implemented using a Deep Neural Network (DNN). The model is designed to solve the multi-label classification problem which predicts email categories for three types simultaneously for each email. The TensorFlow Keras API is used. Given a small dataset, only two dense layers with Relu activation and L2 regularization are used in the hidden layers to avoid overfitting, combined with Dropout and BatchNormalization layers. 


The output layer is a dense layer with units equal to the number of email categories in all types; a sigmoid activation is used to output probability for each category. The model is compiled using the “binary_crossentropy” loss function, which is suitable for multi-label classifications as each label is evaluated separately. The Adam optimizer is selected, and the “binary_accuracy” metric is employed. An early stopping callback is added as well.


The model accuracy is then evaluated using a chained multi-output approach, where the accuracy in a latter Type is dependent on the previous type; it is only accounted for if the previous type is correct. An example output is shown below, with a testing accuracy of 74.47% and training accuracy of 91.7%. To improve the model's performance, more samples are required.
</div>


<div align="center">
<img src="https://github.com/user-attachments/assets/3ddf9957-d4b2-4c49-8813-11e13fb92592"  width="900" />
</div>
