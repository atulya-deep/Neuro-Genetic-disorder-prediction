# Neuro-Genetic-disorder-prediction
Alzheimer’s and Parkinson’s disease are the most common forms of dementia that degenerate neurons in the brain cells. 
This paper targets a comparative study on the performance of machine learning classifier and Neural Network techniques in neuro-degenerative data. The Neural Network algorithms gives classification accuracy ~92% with One hot Encoding Method
![Understanding-Memory-Loss-Alzheimers-disease-the-basics jpg](https://user-images.githubusercontent.com/83969166/211157474-d23f5b30-2607-4886-b185-5d933377aa61.png)
# DATASET
```python
The dataset we are using for our project is the ADPD dataset also known as Alzheimer's and Parkinson's disease dataset.It has 1439 attributes and 199 entries. 
The first attribute represents the gene names and the following attributes represent the gene experimental values of patients. 
The dataset was made from a congregation of multiple gene based datasets including Common Genes Alzheimer Parkinson(CGAP), brain tumore and glioblastoma. 
All these datasets were downloaded from Biolabs Data Set Repository which stores both experimental values and gene names.
```
# MODEL
```python
______________________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 64)                92032     
                                                                 
 leaky_re_lu (LeakyReLU)     (None, 64)                0         
                                                                 
 dense_1 (Dense)             (None, 128)               8320      
                                                                 
 leaky_re_lu_1 (LeakyReLU)   (None, 128)               0         
                                                                 
 dense_2 (Dense)             (None, 256)               33024     
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense_3 (Dense)             (None, 128)               32896     
                                                                 
 leaky_re_lu_2 (LeakyReLU)   (None, 128)               0         
                                                                 
 dense_4 (Dense)             (None, 64)                8256      
                                                                 
 dense_5 (Dense)             (None, 3)                 195       
                                                                 
=================================================================
Total params: 174,723
Trainable params: 174,723
Non-trainable params: 0
```
