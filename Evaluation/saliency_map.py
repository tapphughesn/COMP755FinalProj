import os
from glob import glob
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def preprocess(image):
    transform = T.Compose([
        T.Resize((244, 244)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return transform(image)


def plot_saliency(X, saliency):
    X = X.reshape(-1, 244, 244)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(X.cpu().detach().numpy().transpose(1, 2, 0))
    ax[0].axis("off")
    ax[1].imshow(saliency.cpu(), cmap="hot")
    ax[1].axis("off")
    plt.tight_layout()
    fig.suptitle("The image and its Saliency Map")
    plt.show()


# Initializing model
model = torchvision.models.vgg19(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Running the model in evaluation form
model.eval()

# Opening the images
img_dir = "CelebAHQ256_cleaned"
img_dir_inpainted = "CelebAHQ256_inpainted_gatedconv"
images = sorted(glob(os.path.join(img_dir, '*.jpg')))
images_inpainted = sorted(glob(os.path.join(img_dir_inpainted, '*.png')))

for j in range(len(images)):
    # Reading an image
    image = Image.open(images[j])
    image_i = Image.open(images_inpainted[j])
    X = torch.unsqueeze(preprocess(image), 0)
    X_i = torch.unsqueeze(preprocess(image_i), 0)

    # Finding the gradient of the image
    X = X.to(device)
    X_i = X_i.to(device)
    X.requires_grad_()
    X_i.requires_grad_()

    # Getting the scores
    scores = model(X)
    scores_i = model(X_i)
    score_max_index = scores.argmax()
    score_max_index_i = scores_i.argmax()
    score_max = scores[0, score_max_index]
    score_max_i = scores_i[0, score_max_index_i]

    # Getting the derivative of the outputs/scores
    score_max.backward()
    score_max_i.backward()

    saliency, _ = torch.max(X.grad.data.abs(), dim=1)
    saliency_i, _ = torch.max(X_i.grad.data.abs(), dim=1)
    resize = T.Resize((256, 256))
    saliency = resize(saliency)
    saliency_i = resize(saliency_i)
    saliency = saliency.reshape(256, 256)
    saliency_i = saliency_i.reshape(256, 256)

    # Plotting the image and saliency map
    # plot_saliency(X, saliency)
    # plot_saliency(X_i, saliency_i)

    # Saving the saliency images
    out_dir = "saliency_original/"
    out_name = images[j][len(img_dir)+1:-4]
    out_dir_i = "saliency_inpainted/"
    out_name_i = images_inpainted[j][len(img_dir_inpainted)+1:-4]
    np.save(out_dir+out_name, saliency.numpy())
    np.save(out_dir_i+out_name_i, saliency_i.numpy())

    # Testing if the data was alright
    a = np.load(out_dir_i+out_name_i+".npy")
    print(a.shape)
