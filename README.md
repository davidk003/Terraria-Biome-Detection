# Terraria-Biome-Detection
## Data Collection

Done via a custom Terraria mod for collecting in-game cleaned screenshots:

[Terraria-Biome-Dataset-Collector](https://github.com/davidk003/Terraria-Biome-Dataset-Collector)

[Supplementary Video Demo](https://drive.google.com/file/d/1G6yc_C0iWBHlsKakjIjTHp3ytCM14XDE/view?usp=sharing)



Files

RESNET18/src/dataprep.py
 - Initializes various arrays and functions that transform dataset images

RESNET18/src/resnet18/terraria_data.py
 - Implements Terraria Dataset class and dataloader functions.

RESNET18/src/resnet18/resnet_model.py
 - Initializes resnet model with output space reduced to 13 classes

RESNET18/src/resnet18/resnet_terraria.ipynb
 - Contains pipeline from initializations to training, then exporting model to .onnx

RESNET18/src/resnet18/resnet18_onnx_validation.ipynb
 - Contains .onnx model validation including confusion matrix

RESNET18/src/resnet18/resnet18_realtime_inference.ipynb
 - Contains real-time inference program

RESNET18/src/resnet18/checkpoints
 - Path that model .pth and .onnx files output to from resnet_terraria.ipynb. Currently empty due to filesize, files linked in RESNET18/Readme.md
