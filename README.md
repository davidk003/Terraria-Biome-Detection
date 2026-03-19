# Terraria-Biome-Detection
## Data Collection

Done via a custom Terraria mod for collecting in-game cleaned screenshots:

[Terraria-Biome-Dataset-Collector](https://github.com/davidk003/Terraria-Biome-Dataset-Collector)

[Supplementary Video Demo](https://drive.google.com/file/d/1G6yc_C0iWBHlsKakjIjTHp3ytCM14XDE/view?usp=sharing)

[Uploaded kaggle dataset](https://www.kaggle.com/datasets/davidkim2003/terraria-screenshots-for-biome-classification)



Files

`RESNET18/src/dataprep.py`
 - Defines the image settings and preprocessing steps used to prepare screenshots for training and validation.

`RESNET18/src/resnet18/terraria_data.py` AND `src/efficientnetv2/terraria_data.py`
 - Loads the Terraria screenshot dataset and sets up the dataloaders used during model training.

`RESNET18/src/resnet18/resnet_model.py`
 - Initializes resnet model with output space reduced to 13 classes

`RESNET18/src/resnet18/resnet_terraria.ipynb`
 - Contains pipeline from initializations to training, then exporting model to .onnx

`RESNET18/src/resnet18/resnet18_onnx_validation.ipynb`
 - Contains .onnx model validation including confusion matrix

`RESNET18/src/resnet18/resnet18_realtime_inference.ipynb`
 - Contains real-time inference program

`RESNET18/src/resnet18/checkpoints`
 - Path that model .pth and .onnx files output to from resnet_terraria.ipynb. Currently empty due to filesize, files linked in RESNET18/Readme.md

`src/efficientnetv2/e-netv2-terraria-pretrained.ipynb`
 - Contains the EfficientNetV2 training pipeline using pretrained weights, including dataset setup, training, evaluation, and saving the trained model.

 `src/efficientnetv2/e-netv2-terraria-pretrained.ipynb`
 - pdf version of `src/efficientnetv2/e-netv2-terraria-pretrained.ipynb` for better notebook viewing.

`src/efficientnetv2/e-netv2-terraria-not-pretrained.ipynb`
 - Contains the EfficientNetV2 training pipeline without pretrained weights, including dataset setup, training, evaluation, and saving the trained model.

`realtime_inference_fullframe_min_opencv.ipynb`
 - Contains a real-time inference program that captures the full screen, runs biome prediction on each frame, and displays the predicted biome live.
