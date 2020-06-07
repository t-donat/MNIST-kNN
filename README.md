# MNIST-kNN
This machine learning algorithm that uses kNN to classify the NMIST handwritten images as well as free spoken digits. The data can be preprocessed before classification: For images, a PCA-transform is calculated, and for audio samples the DFT is calculated.

## Helper
This class contains staticmethods which perform type checks and some value checks. It gets imported in all other files.

## Dataset

The Dataset class is used to store the data samples. Running the appropriate `create_ds_image(<path>)` or `create_ds_audio(<path>)` will create a Dataset Object with all .png or .wav files in the directory passed as an argument.
### Dataset methods
#### `generate_sets(method=random,num_samples=None)`: 
Creates a training and test set split by the ratio set by the property `train_size` (which is set to 0.8 by default). Choose from 'random' (default), 'equal' and 'stratified': 
* `random`: randomly choose the samples out of all of them
* `equal`: attempt to take an equal ammount of samples per digit
* `stratified`: attempt to keep the ratios of different digits to eathother in the dataset

Both equal and stratified will output a warning if returning exactly num_samples many features isn't possible

returns a tuple of tuples: ((<train features matrix>,<train labels>),(<test features matrix>,<test labels>)

#### `visualize(sample_indices)`
Visualizes the sample(s) - argument can be single index or iterable of indexes.

#### `table(num_samples=None)`
Create a table with nuim_samples many entries that lists the sample name, its label and the dtype

## Classification
Class used to actually classify the data sets. Contains the following staticmethods:
#### `classify_nn(train_features,train_labels,test_features,k=1)`
Uses the cosine distance to determine the k closest matches for any given test feature. The most common label out of these k nearst is chosen as the prediction. Should there me multiple most common labels, of these the ones with the lower cosine distance dis chosen.

#### `accuracy_scores(true_labels,predicted_labels)`
Compares the correct labels to the predicted ones and calculates the accuracy.
Returns a tuple, the first value is the global accuracy, the second is a list of accuracies for every individual digit

#### `confusion_matrix(true_labels,predicted_labels)`
