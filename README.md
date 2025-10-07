# Garbage Sorter Using Resnet- 18
This project trains a ResNet-18 image classification model on a garbage classification dataset and deploys it to a Jetson Nano for real-time garbage sorting.
The model classifies waste into multiple categories such as battery, cardboard, glass, paper, biological, clothes, metal, plastic, shoes, trash, etc.

# Dataset 
We use the Garbage Classification v2 dataset available on Kaggle:
 https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2/data
Download Instructions
Create a free Kaggle account (if you don’t have one).
Go to the dataset page.
Click Download to get archive.zip (~19,773 images).
Move archive.zip to your Google Drive so you can access it from Google Colab.
eployment: Jetson Nano for inference
# Training on Google Colab
Mount Google Drive to access the dataset:from google.colab import drive
drive.mount('/content/drive')
Unzip dataset into Colab local storage:!unzip -qo "/content/drive/MyDrive/archive.zip" -d "/content/"
The dataset should now be in /content/garbage-dataset.
Set the dataset directory:DATA_DIR = "/content/garbage-dataset"
Ensure your DataLoaders use num_workers=0 in Colab to avoid hanging:train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
Train the model:num_epochs = 10
# training loop here
The script saves the best model as:best_resnet18_garbage.pth
Download the trained model to your computer:from google.colab import files
files.download("best_resnet18_garbage.pth")
# Deploying to Jetson Nano
1. Transfer the Model
Use scp from your Mac/Linux terminal to copy the model to the Jetson Nano:

scp best_resnet18_garbage.pth nvidia@<JETSON_IP>:/home/nvidia/
(Replace nvidia and <JETSON_IP> with your Jetson’s username and IP.)
If you don’t know the IP, scan your network (e.g., with nmap -sn 192.168.x.0/24) or check your router’s connected devices list.

2. Load the Model on Jetson Nano

3. Run Inference
For single image prediction:

from PIL import Image
from torchvision import transforms

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_t = val_transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_t)
        pred = output.argmax(1).item()
    return pred

print(predict_image("test_image.jpg"))
