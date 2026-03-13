RESNET 18 Biome Classifier + more

Dataset would sit in RESNET18/data/
  Starting with ~1500 images, cropping 9 images from center of screen to expand set for training

Realtime inference program
  Take crop of center of screen for realtime probability of biome based on pretrained model
  Top 3 candidates displayed with their confidence/probability displayed

Model would be saved in **/checkpoints as .onnx file. Too large for web upload so will put in gdrive.

