# Hate-Post-detection

### ENVIRONMENT CONFIGURATION 

Requirements:

To replicate the experiments,  environments are required.

```conda create --name <env> --file environments/requirements_fusion.txt```

#### DATASET DOWNLOAD
For the FBHM dataset, The data may be distributed upon request and for academic purposes only. To request the datasets, please fill out the following form: https://forms.gle/AGWMiGicBHiQx4q98
After submitting the required info, participants will have a link to a folder containing the datasets in a zip format (train, training and development) and the password to uncompress the files.
After downloading the images, organize it as follows:

```
└───datasets
     └───FB
         └───files
         └───img
└───models
    └───roberta_resnet
        └───saved
        └───Preds
        └───dataloader_adv_train.py
        └───robresgat_fb4.py
        └───testing.py
```

#### UTILS

The utils folder contains the code to process the data from the FBHM dataset to be easily used for training, validation and testing.
The dataset folder already contains the JSON files for the train, test and validation set.
Download the images from the dataset and place it in the 'dataset' directory.


#### PRE-RUN

Our model will be provided upon request for testing and evaluation.
After downloading the model,save it in the "saved" folder following the file structure below
```
└───saved
     └───fb_weighted_best81.53.pth
└───dataloader_adv_train.py
└───Preds
└───robresgat_fb4.py
└───testing.py
```

#### RUN

Run the robresgat_fb4.py file to execute the project.
Run the testing.py file to test the model in the saved path.
Run the make_jsons_fb.py to create json files for the train, validation and test set of the dataset.

#### RESULTS

The preds folder contains the predictions of our model on the test set for human level judgment.
The report folder contains the report of training the model as well as its perfomance on the validation dataset.
