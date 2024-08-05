# Domain Discrimination Expert Weighted Network for Multi-Source Carotid Artery Plaque Classification

Our method adds a domain discriminator expert to classify echo patterns in multi-source carotid plaque ultrasound data. The specific installation steps and usage instructions are detailed below.

![Untitled](Domain%20Discrimination%20Expert%20Weighted%20Network%20for%20%20f53768450b49415a84cad9fb3df9c955/Untitled.png)

### Requirements

torch = 2.2.2+cu121
torchmetrics = 1.3.2

### Usage

The root path is used to store the root directory of the dataset and the default directory is './'
Source1 is the dataset directory name of source dataset 1.
Source2 is the dataset directory name of source dataset 2.
Target is the dataset directory name of target dataset.

```
  python wucf.py --source1 [source1 name] --source2 [source2 name] --target [target name] --rootpath [rootpath]
```

### Results

The evaluation metrics are recall, precision, and accuracy. All results, including various losses, will be stored in the "record" folder.
