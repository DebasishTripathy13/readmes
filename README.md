```python
Initialize 2xT4 GPUs
We will use both Kaggle T4 GPUs and we will use mixed precision.

import os, gc
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import tensorflow as tf
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
print('TensorFlow version =',tf.__version__)

# USE MULTIPLE GPUS
gpus = tf.config.list_physical_devices('GPU')
if len(gpus)<=1: 
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print(f'Using {len(gpus)} GPU')
else: 
    strategy = tf.distribute.MirroredStrategy()
    print(f'Using {len(gpus)} GPUs')
```
This section initializes the GPU environment by setting the visible CUDA devices and importing necessary libraries. It checks the number of available GPUs and sets up a distribution strategy accordingly (using either a single GPU or mirrored strategy for multiple GPUs).

```python
VER = 5

# IF THIS EQUALS NONE, THEN WE TRAIN NEW MODELS
# IF THIS EQUALS DISK PATH, THEN WE LOAD PREVIOUSLY TRAINED MODELS
LOAD_MODELS_FROM = '/kaggle/input/brain-efficientnet-models-v3-v4-v5/'

USE_KAGGLE_SPECTROGRAMS = True
USE_EEG_SPECTROGRAMS = True
```
Here, we set the version number (`VER`) to 5 and defines the path to load pre-trained models (`LOAD_MODELS_FROM`). The flags `USE_KAGGLE_SPECTROGRAMS` and `USE_EEG_SPECTROGRAMS` are set to `True` to use both types of spectrograms for training.

```python
# USE MIXED PRECISION
MIX = True
if MIX:
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
    print('Mixed precision enabled')
else:
    print('Using full precision')
```
This code block enables mixed precision training if `MIX` is set to `True`, which can speed up computations on compatible hardware. Otherwise, it uses full precision.

```python
Load Train Data
df = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/train.csv')
TARGETS = df.columns[-6:]
print('Train shape:', df.shape )
print('Targets', list(TARGETS))
df.head()
```
This section loads the training data from a CSV file and defines the target variables (`TARGETS`). It prints the shape of the training data and the list of target variables. The `df.head()` function is called to display the first few rows of the training data.

```python
Create Non-Overlapping Eeg Id Train Data
The competition data description says that test data does not have multiple crops from the same eeg_id. Therefore we will train and validate using only 1 crop per eeg_id. There is a discussion about this here.

train = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
    {'spectrogram_id':'first','spectrogram_label_offset_seconds':'min'})
train.columns = ['spec_id','min']

tmp = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
    {'spectrogram_label_offset_seconds':'max'})
train['max'] = tmp

tmp = df.groupby('eeg_id')[['patient_id']].agg('first')
train['patient_id'] = tmp

tmp = df.groupby('eeg_id')[TARGETS].agg('sum')
for t in TARGETS:
    train[t] = tmp[t].values
    
y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1,keepdims=True)
train[TARGETS] = y_data

tmp = df.groupby('eeg_id')[['expert_consensus']].agg('first')
train['target'] = tmp

train = train.reset_index()
print('Train non-overlapp eeg_id shape:', train.shape )
train.head()
```
This section creates a new train dataset with non-overlapping EEG IDs. Since the test data does not have multiple crops from the same EEG ID, the author filters the training data to have only one crop per EEG ID. This is done by grouping the data by 'eeg_id' and aggregating various columns using different functions ('first', 'min', 'max', 'sum'). The target variables are also normalized to sum up to 1. Finally, the 'expert_consensus' column is added, and the new dataset is reset with a new index.

```python
Read Train Spectrograms
First we need to read in all 11k train spectrogram files. Reading thousands of files takes 11 minutes with Pandas. Instead, we can read 1 file from my Kaggle dataset here which contains all the 11k spectrograms in less than 1 minute! To use my Kaggle dataset, set variable READ_SPEC_FILES = False. Thank you for upvoting my helpful dataset :-)

%%time
READ_SPEC_FILES = False

# READ ALL SPECTROGRAMS
PATH = '/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/'
files = os.listdir(PATH)
print(f'There are {len(files)} spectrogram parquets')

if READ_SPEC_FILES:    
    spectrograms = {}
    for i,f in enumerate(files):
        if i%100==0: print(i,', ',end='')
        tmp = pd.read_parquet(f'{PATH}{f}')
        name = int(f.split('.')[0])
        spectrograms[name] = tmp.iloc[:,1:].values
else:
    spectrograms = np.load('/kaggle/input/brain-spectrograms/specs.npy',allow_pickle=True).item()
```
This section reads the train spectrograms from the provided data files. Instead of reading thousands of individual files, which can take a long time, the author suggests using a pre-loaded Kaggle dataset containing all the spectrograms in a single file. By setting `READ_SPEC_FILES` to `False`, the spectrograms are loaded from the provided Kaggle dataset, which is much faster.

```python
Read EEG Spectrograms
In version 4 onward, we use EEG spectrograms in addition to Kaggle spectrograms. The EEG spectrograms come from my Kaggle dataset here (which were created from my spectrogram starter here). Thank you for upvoting my Kaggle dataset!

%%time
READ_EEG_SPEC_FILES = False

if READ_EEG_SPEC_FILES:
    all_eegs = {}
    for i,e in enumerate(train.eeg_id.values):
        if i%100==0: print(i,', ',end='')
        x = np.load(f'/kaggle/input/brain-eeg-spectrograms/EEG_Spectrograms/{e}.npy')
        all_eegs[e] = x
else:
    all_eegs = np.load('/kaggle/input/brain-eeg-spectrograms/eeg_specs.npy',allow_pickle=True).item()
```
Starting from version 4, the notebook also uses EEG spectrograms in addition to Kaggle spectrograms. This section reads the EEG spectrograms from a Kaggle dataset provided by the author. Similar to the previous section, the author provides an option to read individual EEG spectrogram files (`READ_EEG_SPEC_FILES=True`) or load them from a pre-loaded Kaggle dataset (`READ_EEG_SPEC_FILES=False`), with the latter being much faster.

```python
Train DataLoader
This dataloader outputs 4 spectrogram images as a 4 channel image of size 128x256x4 per train sample. This notebook version is not using data augmention but the code is available below to experiment with albumentations data augmention. Just add augment = True when creating the train data loader. And consider adding new transformations to the augment function below.

UPDATE: In version 4 onward, our dataloader outputs both Kaggle spectrograms and EEG spectrogams as 8 channel image of size 128x256x8.

import albumentations as albu
TARS = {'Seizure':0, 'LPD':1, 'GPD':2, 'LRDA':3, 'GRDA':4, 'Other':5}
TARS2 = {x:y for y,x in TARS.items()}

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, batch_size=32, shuffle=False, augment=False, mode='train',
                 specs = spectrograms, eeg_specs = all_eegs): 

        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.mode = mode
        self.specs = specs
        self.eeg_specs = eeg_specs
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        ct = int( np.ceil( len(self.data) / self.batch_size ) )
        return ct

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)
        if self.augment: X = self.__augment_batch(X) 
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange( len(self.data) )
        if self.shuffle: np.random.shuffle(self.indexes)
                        
    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' 
        
        X = np.zeros((len(indexes),128,256,8),dtype='float32')
        y = np.zeros((len(indexes),6),dtype='float32')
        img = np.ones((128,256),dtype='float32')
        
        for j,i in enumerate(indexes):
            row = self.data.iloc[i]
            if self.mode=='test': 
                r = 0
            else: 
                r = int( (row['min'] + row['max'])//4 )

            for k in range(4):
                # EXTRACT 300 ROWS OF SPECTROGRAM
                img = self.specs[row.spec_id][r:r+300,k*100:(k+1)*100].T
                
                # LOG TRANSFORM SPECTROGRAM
                img = np.clip(img,np.exp(-4),np.exp(8))
                img = np.log(img)
                
                # STANDARDIZE PER IMAGE
                ep = 1e-6
                m = np.nanmean(img.flatten())
                s = np.nanstd(img.flatten())
                img = (img-m)/(s+ep)
                img = np.nan_to_num(img, nan=0.0)
                
                # CROP TO 256 TIME STEPS
                X[j,14:-14,:,k] = img[:,22:-22] / 2.0
        
            # EEG SPECTROGRAMS
            img = self.eeg_specs[row.eeg_id]
            X[j,:,:,4:] = img
                
            if self.mode!='test':
                y[j,] = row[TARGETS]
            
        return X,y
    
    def __random_transform(self, img):
        composition = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            #albu.CoarseDropout(max_holes=8,max_height=32,max_width=32,fill_value=0,p=0.5),
        ])
        return composition(image=img)['image']
            
    def __augment_batch(self, img_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i, ] = self.__random_transform(img_batch[i, ])
        return img_batch
```
This section defines a custom data generator class `DataGenerator` that inherits from `tf.keras.utils.Sequence`. The data generator is responsible for loading and preprocessing the spectrogram and EEG data. The key points are:

- The data generator outputs a 4-channel image of size 128x256x4 for each sample, where each channel represents a spectrogram image.
- Starting from version 4, the data generator outputs an 8-channel image of size 128x256x8, with the first 4 channels representing Kaggle spectrograms and the last 4 channels representing EEG spectrograms.
- The class contains methods for data generation, shuffling, and optional data augmentation using the Albumentations library.
- The `__data_generation` method performs preprocessing steps such as extracting spectrogram rows, log transformation, standardization, and cropping.
- The `__random_transform` and `__augment_batch` methods provide functionality for data augmentation, although they are not used in the current version of the notebook.

```python
Display DataLoader
Below we display example dataloader spectrogram images.

gen = DataGenerator(train, batch_size=32, shuffle=False)
ROWS=2; COLS=3; BATCHES=2

for i,(x,y) in enumerate(gen):
    plt.figure(figsize=(20,8))
    for j in range(ROWS):
        for k in range(COLS):
            plt.subplot(ROWS,COLS,j*COLS+k+1)
            t = y[j*COLS+k]
            img = x[j*COLS+k,:,:,0][::-1,]
            mn = img.flatten().min()
            mx = img.flatten().max()
            img = (img-mn)/(mx-mn)
            plt.imshow(img)
            tars = f'[{t[0]:0.2f}'
            for s in t[1:]: tars += f', {s:0.2f}'
            eeg = train.eeg_id.values[i*32+j*COLS+k]
            plt.title(f'EEG = {eeg}\nTarget = {tars}',size=12)
            plt.yticks([])
            plt.ylabel('Frequencies (Hz)',size=14)
            plt.xlabel('Time (sec)',size=16)
    plt.show()
    if i==BATCHES-1: break
```
This section displays example spectrogram images generated by the `DataGenerator`. It creates an instance of the `DataGenerator` with the train data and iterates over batches of data. For each batch, it plots a grid of spectrogram images, along with their corresponding EEG IDs and target values. The plots are displayed using `plt.show()`.

```python
Train Scheduler
We will train our model with a Step Train Schedule for 4 epochs. First 2 epochs are LR=1e-3. Then epochs 3 and 4 use LR=1e-4 and 1e-5 respectively. (Below we also provide a Cosine Train Schedule if you want to experiment with it. Note it is not used in this notebook).

# STEP SCHEDULE
class LR(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        if epoch == 2: 
            print('LEARNING RATE FROM 1E-3 to 1E-4')
            self.model.optimizer.learning_rate = 1e-4
        if epoch == 3: 
            print('LEARNING RATE FROM 1E-4 to 1E-5')
            self.model.optimizer.learning_rate = 1e-5
            
LR = LR()

# COSINE SCHEDULE
def scheduler(epoch):
    if epoch < 2:
        return 1e-3
    else:
        return 1e-3 * tf.math.cos(np.pi*epoch/4) 

# set the `lr` attribute of the optimizer
# before compiling and running training
sched = tf.keras.callbacks.LearningRateScheduler(scheduler)
```
This section defines a learning rate scheduling strategy for training the model. The provided code includes two options:

1. **Step Schedule**: This schedule is implemented through the `LR` callback class. It starts with a learning rate of 1e-3 for the first two epochs, then decreases the learning rate to 1e-4 for the third epoch, and finally to 1e-5 for the fourth epoch.

2. **Cosine Schedule**: This schedule is defined by the `scheduler` function, which follows a cosine annealing schedule. The learning rate starts at 1e-3 for the first two epochs, and then decreases according to a cosine function for the remaining epochs.

The code creates an instance of the `LR` callback class, which will be used during model training to update the learning rate according to the step schedule.

```python
Build EfficientNet Model
Version 1-3 uses EfficientNet B2. Version 4 uses EfficientNet B0. Our models receives both Kaggle spectrograms and EEG spectrograms from our data loader. We then reshape these 8 spectrograms into 1 large flat image and feed it into EfficientNet.

!pip install --no-index --find-links=/kaggle/input/tf-efficientnet-whl-files /kaggle/input/tf-efficientnet-whl-files/efficientnet-1.1.1-py3-none-any.whl
import efficientnet.tfkeras as efn

def build_model():
    
    inp = tf.keras.Input(shape=(128,256,8))
    base_model = efn.EfficientNetB0(include_top=False, weights=None, input_shape=None)
    base_model.load_weights('/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5')
    
    # RESHAPE INPUT 128x256x8 => 512x512x3 MONOTONE IMAGE
    # KAGGLE SPECTROGRAMS
    x1 = [inp[:,:,:,i:i+1] for i in range(4)]
    x1 = tf.keras.layers.Concatenate(axis=1)(x1)
    # EEG SPECTROGRAMS
    x2 = [inp[:,:,:,i+4:i+5] for i in range(4)]
    x2 = tf.keras.layers.Concatenate(axis=1)(x2)
    # MAKE 512X512X3
    if USE_KAGGLE_SPECTROGRAMS & USE_EEG_SPECTROGRAMS:
        x = tf.keras.layers.Concatenate(axis=2)([x1,x2])
    elif USE_EEG_SPECTROGRAMS: x = x2
    else: x = x1
    x = tf.keras.layers.Concatenate(axis=3)([x,x,x])
    
    # OUTPUT
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(6,activation='softmax', dtype='float32')(x)
        
    # COMPILE MODEL
    model = tf.keras.Model(inputs=inp, outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate = 1e-3)
    loss = tf.keras.losses.KLDivergence()

    model.compile(loss=loss, optimizer = opt) 
        
    return model
```
This section defines the `build_model` function, which creates an EfficientNet model for the brain activity classification task. Here's a breakdown of the steps:

1. The function installs the required `efficientnet` package from a provided wheel file.
2. It creates a Keras input layer with a shape of (128, 256, 8), expecting an 8-channel image as input.
3. An EfficientNetB0 model is instantiated without the top layers, and pre-trained weights are loaded from a provided file.
4. The input 8-channel image is reshaped into a single 512x512x3 image by concatenating the Kaggle spectrograms and EEG spectrograms along different axes.
5. The reshaped image is passed through the EfficientNetB0 base model.
6. The output of the base model is processed using Global Average Pooling and a Dense layer with 6 output units and a softmax activation.
7. The model is compiled with an Adam optimizer with a starting learning rate of 1e-3 and KL Divergence loss.
8. The compiled model is returned.

The `build_model` function allows for flexibility in using either Kaggle spectrograms, EEG spectrograms, or both, depending on the values of `USE_KAGGLE_SPECTROGRAMS` and `USE_EEG_SPECTROGRAMS` flags.

```python
Train Model
We train using Group KFold on patient id. If LOAD_MODELS_FROM = None, then we will train new models in this notebook version. Otherwise we will load saved models from the path LOAD_MODELS_FROM.

from sklearn.model_selection import KFold, GroupKFold
import tensorflow.keras.backend as K, gc

all_oof = []
all_true = []

gkf = GroupKFold(n_splits=5)
for i, (train_index, valid_index) in enumerate(gkf.split(train, train.target, train.patient_id)):  
    
    print('#'*25)
    print(f'### Fold {i+1}')
    
    train_gen = DataGenerator(train.iloc[train_index], shuffle=True, batch_size=32, augment=False)
    valid_gen = DataGenerator(train.iloc[valid_index], shuffle=False, batch_size=64, mode='valid')
    
    print(f'### train size {len(train_index)}, valid size {len(valid_index)}')
    print('#'*25)
    
    K.clear_session()
    with strategy.scope():
        model = build_model()
    if LOAD_MODELS_FROM is None:
        model.fit(train_gen, verbose=1,
              validation_data = valid_gen,
              epochs=EPOCHS, callbacks = [LR])
        model.save_weights(f'EffNet_v{VER}_f{i}.h5')
    else:
        model.load_weights(f'{LOAD_MODELS_FROM}EffNet_v{VER}_f{i}.h5')
        
    oof = model.predict(valid_gen, verbose=1)
    all_oof.append(oof)
    all_true.append(train.iloc[valid_index][TARGETS].values)
    
    del model, oof
    gc.collect()
    
all_oof = np.concatenate(all_oof)
all_true = np.concatenate(all_true)
```
This section implements the training process for the EfficientNet model using Group K-Fold cross-validation on patient IDs. Here's a breakdown of the steps:

1. The code imports necessary modules and creates empty lists to store out-of-fold predictions (`all_oof`) and true labels (`all_true`).
2. A `GroupKFold` object is created with 5 splits.
3. For each fold:
   - Print the fold number.
   - Create data generators for training and validation sets using the `DataGenerator` class.
   - Print the sizes of the training and validation sets.
   - Clear the TensorFlow session.
   - Build the EfficientNet model within the distribution strategy scope.
   - If `LOAD_MODELS_FROM` is None, train the model on the training set using the `fit` method, with the learning rate scheduler callback. Save the model weights to a file.
   - If `LOAD_MODELS_FROM` is not None, load the pre-trained model weights from the specified path.
   - Predict on the validation set using the `predict` method and store the predictions and true labels.
   - Delete the model object and collect garbage.
4. Concatenate all out-of-fold predictions and true labels into numpy arrays `all_oof` and `all_true`.

This cross-validation approach trains multiple EfficientNet models on different folds of the data and stores the predictions and true labels for further evaluation.

```python
CV Score for EfficientNet
This is CV score for our EfficientNet model.

import sys
sys.path.append('/kaggle/input/kaggle-kl-div')
from kaggle_kl_div import score

oof = pd.DataFrame(all_oof.copy())
oof['id'] = np.arange(len(oof))

true = pd.DataFrame(all_true.copy())
true['id'] = np.arange(len(true))

cv = score(solution=true, submission=oof, row_id_column_name='id')
print('CV Score KL-Div for EfficientNetB2 =',cv)
```
This section calculates the cross-validation score for the EfficientNet model using the KL Divergence metric. Here's what it does:

1. Import the `score` function from the `kaggle_kl_div` module, which is likely a custom implementation of the KL Divergence metric used in the competition.
2. Convert the `all_oof` (out-of-fold predictions) and `all_true` (true labels) numpy arrays into pandas DataFrames.
3. Add an 'id' column to both DataFrames to serve as the row identifier.
4. Call the `score` function, passing the true labels DataFrame as `solution`, the out-of-fold predictions DataFrame as `submission`, and specifying the 'id' column as the `row_id_column_name`.
5. Print the calculated cross-validation score.

This step evaluates the performance of the EfficientNet model on the cross-validation folds using the KL Divergence metric.

```python
Infer Test and Create Submission CSV
Below we use our 5 EfficientNet fold models to infer the test data and create a submission.csv file.

del all_eegs, spectrograms; gc.collect()
test = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/test.csv')
print('Test shape',test.shape)
test.head()

# READ ALL SPECTROGRAMS
PATH2 = '/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/'
files2 = os.listdir(PATH2)
print(f'There are {len(files2)} test spectrogram parquets')
    
spectrograms2 = {}
for i,f in enumerate(files2):
    if i%100==0: print(i,', ',end='')
    tmp = pd.read_parquet(f'{PATH2}{f}')
    name = int(f.split('.')[0])
    spectrograms2[name] = tmp.iloc[:,1:].values
    
# RENAME FOR DATALOADER
test = test.rename({'spectrogram_id':'spec_id'},axis=1)

# READ ALL EEG SPECTROGRAMS
PATH2 = '/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/'
DISPLAY = 1
EEG_IDS2 = test.eeg_id.unique()
all_eegs2 = {}

print('Converting Test EEG to Spectrograms...'); print()
for i,eeg_id in enumerate(EEG_IDS2):
        
    # CREATE SPECTROGRAM FROM EEG PARQUET
    img = spectrogram_from_eeg(f'{PATH2}{eeg_id}.parquet', i<DISPLAY)
    all_eegs2[eeg_id] = img

#########################

# INFER EFFICIENTNET ON TEST
preds = []
model = build_model()
test_gen = DataGenerator(test, shuffle=False, batch_size=64, mode='test',
                         specs = spectrograms2, eeg_specs = all_eegs2)

for i in range(5):
    print(f'Fold {i+1}')
    if LOAD_MODELS_FROM:
        model.load_weights(f'{LOAD_MODELS_FROM}EffNet_v{VER}_f{i}.h5')
    else:
        model.load_weights(f'EffNet_v{VER}_f{i}.h5')
    pred = model.predict(test_gen, verbose=1)
    preds.append(pred)
pred = np.mean(preds,axis=0)
print()
print('Test preds shape',pred.shape)

sub = pd.DataFrame({'eeg_id':test.eeg_id.values})
sub[TARGETS] = pred
sub.to_csv('submission.csv',index=False)
print('Submissionn shape',sub.shape)
sub.head()

# SANITY CHECK TO CONFIRM PREDICTIONS SUM TO ONE
sub.iloc[:,-6:].sum(axis=1)
```
This section focuses on inferring the test data using the trained EfficientNet models and creating a submission file. Here's a breakdown of the steps:

1. Delete the previously loaded `all_eegs` and `spectrograms` variables to free up memory.
2. Load the test data from a CSV file and print its shape.
3. Read the test spectrograms and EEG spectrograms from their respective folders.
4. Preprocess the EEG data by converting it to spectrograms using the `spectrogram_from_eeg` function (not shown in the provided code).
5. Create a `DataGenerator` instance for the test data, setting `mode='test'` and providing the test spectrograms and EEG spectrograms.
6. Initialize an empty list `preds` to store the predictions from each fold.
7. For each of the 5 folds:
   - Print the fold number.
   - Load the trained model weights from either the provided path (`LOAD_MODELS_FROM`) or the local file (`EffNet_v{VER}_f{i}.h5`).
   - Predict on the test data using the loaded model and the `predict` method.
   - Append the predictions to the `preds` list.
8. Calculate the mean of the predictions across all folds and store it in `pred`.
9. Print the shape of the final test predictions.
10. Create a submission DataFrame with the 'eeg_id' column and the target variable columns.
11. Save the submission DataFrame to a CSV file named 'submission.csv'.
12. Print the shape of the submission file and display the first few rows.
13. Perform a sanity check to ensure that the predictions for each sample sum up to 1.

This section generates the final submission file by averaging the predictions from the 5 EfficientNet models trained on different folds. The submission file is saved in the required format for the competition.
