import torch
import requests

#! wget https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download -O weights.zip
#! unzip -d weights -j weights.zip
from model import ClipPred
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
from torch.cuda.amp import autocast, GradScaler

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ClipPred(reduce_dim=64,device=device)
model.to(device)
model.eval()
# non-strict, because we only stored decoder weights (not CLIP weights)
path = "weights.pth"
model.load_state_dict(torch.load(path, map_location=torch.device(device)), strict=False)

# load and normalize image
#input_image = Image.open('image1.png')

# or load from URL...
image_url = 'https://farm5.staticflickr.com/4141/4856248695_03475782dc_z.jpg'
input_image = Image.open(requests.get(image_url, stream=True).raw)
#input_image = Image.open('image1.png').convert("RGB")
input_image = Image.open('100.jpg').convert("RGB")
#input_image = Image.open('/mnt/disks/data/images/2359296.jpg').convert("RGB")
#input_image = Image.open('/mnt/disks/data/images/2363368.jpg').convert("RGB")
#input_image = Image.open('/mnt/disks/data/images/2407047.jpg').convert("RGB")
#input_image = Image.open('/mnt/disks/data/images/2824.jpg').convert("RGB")
#input_image = Image.open('/mnt/disks/data/images/361.jpg').convert("RGB")
#input_image = Image.open('/mnt/disks/data/images/568.jpg').convert("RGB")
#input_image = Image.open('/mnt/disks/data/images/2403660.jpg').convert("RGB")
input_image = Image.open('/mnt/disks/data/images/1325.jpg').convert("RGB")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((352, 352)),
])
img = transform(input_image).unsqueeze(0)

#prompts = ['car on left side','car on right side']
prompts = ['a monitor','a remote']
#prompts = ['windows','brown lampshade','bed']
#prompts = ['pink hoodie','blue carpet','blue shirt','standing guy']
#prompts = ['opened laptop','black back','red wall']
#prompts = ['yellow box','light on store front','lamp','black sky above lamp','white car on street',
#        'white car']
#prompts = ['red brick side walk','street','red sign']
#prompts = ['a chair']
#prompts = ['A man on skateboard']
prompts = ['eiffel tower']
N = len(prompts)

#prompts = ['a glass', 'something to fill', 'wood', 'a jar']

# predict
with torch.no_grad():
    data = img.repeat(N,1,1,1).to(device)
    with autocast():
      preds = model(data, prompts)

outputs = [torch.sigmoid(preds[i]).cpu()[0].numpy() for i in range(N)]
threshold, upper, lower = 0.2, 1.0, 0.0

outputs_th = []

for i in range(N):
  a = outputs[i].copy()
  a[a>threshold] = upper
  a[a<threshold] = lower
  outputs_th.append(a)

# visualize prediction
_, ax = plt.subplots(1, N+1, figsize=(8,8))
[a.axis('off') for a in ax.flatten()]
ax[0].imshow(input_image.resize((352,352)))
[ax[i+1].imshow(outputs[i]) for i in range(N)]
[ax[i+1].text(0, -15, prompts[i]) for i in range(N)]
plt.savefig('results.png')
