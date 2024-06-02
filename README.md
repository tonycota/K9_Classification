# K9_Classification

# Image Classification 

Within this section of the repo you will find numerous python scripts analyzing, standardizing and building Convulutional Neural Network Models
based on a dataset of images of dog breeds. 

### Dependencies utilized:
* Tensorflow
* Scikit-Learn
* NumPy
* Matplotlib
* Seaborn
* Pandas
* SQL and SQLite

### Implementation

Since the Dataset containing images of assorted dog breeds was already cropped and cleaned (Resources/cropped), a lot of the data preprocessing time was reduced. However, implementing a CNN model and achieving an accuracy of 75% on the given data was a challenge. Initially achieving a Validation accuracy score of 13%, a different approach was considered; thus the decision of integrating the xception model was made and immediately made a significant impact on Validation accuracy even with just ten epochs while training the model (score of 24%). The script with the xception model applied can be found within the same notebook mentioned before. 

With that, a final approach was attempted. Implementing Standard Scaling to the model. While not common in CNN models, an attempt to optimized the personal model was attempted and achieved even less considerable results, with a validation accuracy of less than 1%. 

## Analyzing Keras Applications Pre-Built Models

After studying different models from the given url (https://keras.io/api/applications/), the project took a turn to see which pre-trained model from ImageNet dataset could classify the images in our dataset best. 3 sample photos were chosen of a Beagle, a Yorkshire Terrier, and a Golden Retriever. 

### Pre-Trained Models Analyzed

* EfficientNetB7
* InceptionsResNetV2
* MobileNet
* VGG19
* Xception

![alt text](Outputs/vgg19_model_screenshot.jpeg)

Given the three sample photos to each pre-trained models, prediction scores of each breed of dog were recorded and ingested into Pandas Dataframes which can be found under the Notebooks folder (efficient_net_b7_model.ipynb, inception_res_net_v2.ipynb, mobile_net_model.ipynb, vgg19_model.ipynb, and xception_model.ipynb). 5 dataframes were written to csv files and were then initialized into a SQLite database (found in joining_tables.ipynb). The dataframes were then joined together to draw conclusive results as to which pre-trained model worked best with our given data. 

![alt text](Outputs/accuracy_graph.png)
Noted, the MobileNet model performed best on the given photo of just the yorkshire terrier, while the VGG19 model predicted the photo second best out of the given dataset. 

![alt text](Outputs/overall_accuracy_graph.png)

In the graph above lists the average prediction accuracies achieved by each pre-trained model. MobileNet performed the overall best with predicting which dog breed was which, with InceptionResNetV2 second and VGG19 third. 

This part of the project was completed to conclude if our given data needed further cleaning in order to build a substantial CNN model. Given these results listed above, it is apparent our data was readily produced. 

Sources
https://www.kaggle.com/code/nayanack/breed-classification/input
https://keras.io/api/applications/
https://stackoverflow.com/questions/69114904/how-to-properly-load-and-use-pre-trained-model-in-keras
Ashley Nguyen's and Armando Cota's submission for Final Project