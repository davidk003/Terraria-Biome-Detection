RESNET 18 Biome Classifier + more

Dataset would sit in RESNET18/data/
  Starting with ~1500 images, cropping 9 images from center of screen to expand set for training

Realtime inference program
  Take crop of center of screen for realtime probability of biome based on pretrained model
  Top 3 candidates displayed with their confidence/probability displayed

Model would be saved in **/checkpoints as .onnx file. Too large for web upload so will put in gdrive.

  [.pth model link "best_resnet18.pth"](https://drive.google.com/file/d/1Clyh9ssS15cLoDfAI3TujJRiKR5J5Kmm/view?usp=drive_link)
  
  [.onnx model link "resnet18_terraria.onnx"](https://drive.google.com/file/d/1qgyukp1wNk_dl3mPhkt5q4D2Mm0dDr0w/view?usp=sharing)


Sample videos of realtime predictions at following link:

 - [Google Drive - 3 videos](https://drive.google.com/drive/folders/1X6qat17HdknNyPwLWKGZDuBbhCRLOMdu?usp=sharing)
    These videos show the full game screen, with a small window showing the cropped sample that is fed into the model, and the top 3 predictions from the model in order.
   
