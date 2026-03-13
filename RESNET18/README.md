RESNET 18 Biome Classifier + more

Dataset would sit in RESNET18/data/
  Starting with ~1500 images, cropping 9 images from center of screen to expand set for training

Realtime inference program
  Take crop of center of screen for realtime probability of biome based on pretrained model
  Top 3 candidates displayed with their confidence/probability displayed

Model would be saved in **/checkpoints as .onnx file. Too large for web upload so will put in gdrive.
  [.pth model link "best_resnet18.pth"](https://drive.google.com/file/d/1Clyh9ssS15cLoDfAI3TujJRiKR5J5Kmm/view?usp=drive_link)
  [.onnx model link "resnet18_terraria.onnx"](https://drive.google.com/file/d/1qgyukp1wNk_dl3mPhkt5q4D2Mm0dDr0w/view?usp=sharing)
