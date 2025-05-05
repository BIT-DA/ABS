# How to Install Datasets

`$DATA` denotes the location where datasets are installed, e.g.

```
$DATA/
|–– office31/
|–– office_home/
|–– visda17/
```

### ImageNet

- Download the dataset from the [official website](https://image-net.org/index.php)
- Create a folder named `imagenet/` under `$DATA`.
- Extract the validation sets to `$DATA/imagenet/val`. The directory structure should look like

```
.
├── ...
├── data                    
│   ├── imagenet            
│   │   ├── val             
│   └── ...                 
└── ...
```

---

### ImageNetV2

- Create a folder named `imagenetv2/` under `$DATA`.
- Go to this github repo [https://github.com/modestyachts/ImageNetV2](https://github.com/modestyachts/ImageNetV2).
- Download the matched-frequency dataset from [https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz](https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz) and extract it to `$DATA/imagenetv2/`.

```
.
├── ...
├── data                    
│   ├── imagenetv2            
│   │   ├── ImageNetV2-matched-frequency/  
│   └── ...                 
└── ...
```

---

### ImageNet-R

- Create a folder named `imagenet-r/` under `$DATA`.
- Download the dataset from [https://github.com/hendrycks/imagenet-r](https://github.com/hendrycks/imagenet-r).
- Extract it to `$DATA/imagenet-r/`.

The directory structure should look like

```
.
├── ...
├── data
│   ├── imagenet-r
│   │   ├── n01443537/
│   │   ├── ...
│   └── ...
└── ...
```

---

### ImageNet-A

- Create a folder named `imagenet-a/` under `$DATA`.
- Download the dataset from [https://github.com/hendrycks/natural-adv-examples](https://github.com/hendrycks/natural-adv-examples).
- Extract it to `$DATA/imagenet-a/`.

The directory structure should look like

```
.
├── ...
├── data
│   ├── imagenet-a
│   │   ├── n01498041
│   │   ├── ...
│   └── ...
└── ...
```

---

### ImageNet-S

- Create a folder named `imagenet-s/` under `$DATA`.
- Download the dataset from [https://github.com/LUSSeg/ImageNet-S](https://github.com/LUSSeg/ImageNet-S).
- Extract it to `$DATA/imagenet-s/`.

The directory structure should look like

```
.
├── ...
├── data
│   ├── imagenet-s
│   │   ├── n01440764/
│   │   ├── ...
│   └── ...
└── ...
```

---


### CUB-200-2011

- Create a folder named `CUB_200_2011/` under `$DATA`.
- Download the dataset from [https://www.vision.caltech.edu/datasets/cub_200_2011/](https://www.vision.caltech.edu/datasets/cub_200_2011/) and extract it to `$DATA/CUB_200_2011/`.

The directory structure should look like

```
.
├── ...
├── data                    
│   ├── CUB-200-2011            
│   └── ...                 
└── ...
```

### Describable Textures Dataset

- Download the dataset from [https://www.robots.ox.ac.uk/~vgg/data/dtd/](https://www.robots.ox.ac.uk/~vgg/data/dtd/) and extract it to `$DATA/`.

The directory structure should look like

```
.
├── ...
├── data                    
│   ├── dtd     
│   │   ├── images/        
│   │   ├── imdb/        
│   │   ├── labels/        
│   │   ├── ...         
│   └── ...                 
└── ...
```

---

### Food-101

- Download the dataset from [https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) and extract it to `$DATA/`.

The directory structure should look like

```
.
├── ...
├── data                    
│   ├── food-101     
│   │   ├── images/        
│   │   ├── meta/              
│   │   ├── ...         
│   └── ...                 
└── ...
```

### oxford-iiit-pet

- Download the dataset from [https://www.robots.ox.ac.uk/~vgg/data/pets/](https://www.robots.ox.ac.uk/~vgg/data/pets/) and extract it to `$DATA/`.

The directory structure should look like

```
.
├── ...
├── data
│   ├── oxford-iiit-pet
│   │   ├── images/
│   │   ├── annotations/
│   │   ├── ...
│   └── ...
└── ...
```
