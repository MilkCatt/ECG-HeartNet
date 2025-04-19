# ECG-HeartNet

## Data
Get the processed_data.zip from [here](https://seafile.purplecircle.xyz/d/43fa61af9f724398a476/) and unzip it to the project folder.

The folder structure should be arranged like this:

```
.
├── data
│   ├── ecg
│   │   ├── 00001.csv
│   │   ├── ...
│   │   └── 45551.csv
│   ├── ecg_clipped
│   │   ├── 00001.csv
│   │   ├── ...
│   │   └── 45551.csv
│   └── diagnoses_10.csv
│   ...
└── model.ipynb
```

## Codebase Summary
**archive** - This folder contains old code testing files and previous iterations
**data** - This folder contains the 12-lead data (ecg & ecg_clipped) as well as the master sheets containing a directory of the 12-lead data which the model reads from (diagnoses_10.csv, diagnoses_balanced.csv, etc.)
**metadata** - This folder contains the mean and standard deviation of each of the 12-leads.
**scripts** - This folder contains all the scripts used for data preprocessing, cleaning, and visualisation.
**training_progress** - This folder contains model params and histories across the different iterations of our models.
*models.py* - This file contains the model objects used
*objects.py* - This file contains data-related objects
*run.ipynb* - This is the main file which the model can be run from
*utils.py* - This file contains additional helper functions 
