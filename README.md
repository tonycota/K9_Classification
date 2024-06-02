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

Given the three sample photos to each pre-trained models, prediction scores of each breed of dog were recorded and ingested into Pandas Dataframes which can be found under the Notebooks folder (efficient_net_b7_model.ipynb, inception_res_net_v2.ipynb, mobile_net_model.ipynb, vgg19_model.ipynb, and xception_model.ipynb). 5 dataframes were written to csv files and were then initialized into a SQLite database (found in joining_tables.ipynb). The dataframes were then joined together to draw conclusive results as to which pre-trained model worked best with our given data. 
![Image Description]('Outputs/accuracy_graph.png')




Ashley Nguyen's and Armando Cota's submission for Final Project
