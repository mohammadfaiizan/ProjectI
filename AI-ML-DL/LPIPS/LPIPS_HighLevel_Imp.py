'''
The paper does not provide explicit implementation details of the network architectures but mentions using well-known architectures like VGG, AlexNet, and SqueezeNet. The perceptual similarity metric is derived from deep feature activations of these networks.

However, I can summarize the architectural details and provide pseudo-code for implementing perceptual similarity metrics based on the paper.

1. Architectures Used
The following deep networks were used for perceptual similarity:

AlexNet (Krizhevsky et al., 2012)

5 convolutional layers (conv1–conv5)
Fully connected layers (not used for similarity)
ReLU activations
Max-pooling layers
Feature extraction from conv1-conv5
VGG-16 (Simonyan & Zisserman, 2014)

13 convolutional layers + ReLU
Max-pooling layers
Feature extraction from conv1_2, conv2_2, conv3_3, conv4_3, conv5_3
SqueezeNet (Iandola et al., 2017)

1 initial conv layer
Fire modules with 1x1 and 3x3 convolutions
Feature extraction from fire4, fire5, fire9
Self-Supervised and Unsupervised Networks

BiGAN, Split-Brain Autoencoders, K-Means Pretrained Networks
These networks were also used for feature extraction but without traditional supervised training.
2. Computing Perceptual Distance
The perceptual similarity metric is computed as follows:

Extract deep features from selected layers.
Normalize the activations channel-wise.
Compute the L2 distance between corresponding feature activations of the reference and distorted images.
Apply learned linear weights to optimize correlation with human judgments.
3. Code Implementation
a) Load Pretrained Models and Extract Features
Here’s a PyTorch implementation of the perceptual similarity metric:

python
Copy
Edit
'''
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class PerceptualSimilarity(nn.Module):
    def __init__(self, model_name='vgg'):
        super(PerceptualSimilarity, self).__init__()
        self.model_name = model_name

        if model_name == 'vgg':
            model = models.vgg16(pretrained=True).features
            self.layers = ['3', '8', '15', '22', '29']  # conv1_2 to conv5_3
        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True).features
            self.layers = ['0', '3', '6', '8', '10']  # conv1 to conv5
        elif model_name == 'squeezenet':
            model = models.squeezenet1_1(pretrained=True).features
            self.layers = ['2', '5', '8', '11']  # selected fire modules

        self.model = nn.Sequential(*list(model.children())[:max(map(int, self.layers)) + 1])
        self.model.eval()

    def forward(self, x1, x2):
        feats1 = self.extract_features(x1)
        feats2 = self.extract_features(x2)

        # Compute L2 distance
        distances = [torch.norm(f1 - f2, p=2, dim=1).mean() for f1, f2 in zip(feats1, feats2)]
        return sum(distances)

    def extract_features(self, x):
        outputs = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            if str(i) in self.layers:
                outputs.append(x)
        return outputs

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load images
img1 = transform(Image.open("image1.jpg")).unsqueeze(0)
img2 = transform(Image.open("image2.jpg")).unsqueeze(0)

# Compute perceptual similarity
model = PerceptualSimilarity(model_name='vgg')
similarity_score = model(img1, img2)
print(f"Perceptual Similarity Score: {similarity_score.item()}")

'''
b) Fine-Tuning for Perceptual Similarity
The paper also suggests fine-tuning the model using a perceptual similarity dataset (BAPPS). Below is an example of training with a binary classification loss:

python
Copy
Edit
'''
class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(len(feature_extractor.layers), 1)  # Learnable weights per feature

    def forward(self, x1, x2, label):
        distances = torch.stack(self.feature_extractor(x1, x2), dim=1)
        pred = torch.sigmoid(self.fc(distances))  # Predict similarity probability
        loss = nn.BCELoss()(pred, label)  # Binary Cross-Entropy Loss
        return loss

'''
To train:

python
Copy
Edit
'''
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(num_epochs):
    for img1, img2, label in train_loader:
        optimizer.zero_grad()
        loss = model(img1, img2, label)
        loss.backward()
        optimizer.step()

'''
4. Summary of Architectural Insights
Feature Extraction:

Deep networks (VGG, AlexNet, SqueezeNet) extract hierarchical features.
Features are taken from intermediate convolutional layers.
Higher-level features capture perceptual similarity better than pixel-wise metrics.
Distance Computation:

Normalize feature activations.
Compute L2 distance in feature space.
Apply learned weights to align with human perceptual judgments.
Performance Tuning:

Pre-trained networks perform well but fine-tuning on perceptual datasets improves performance.
Even self-supervised and unsupervised models (e.g., BiGAN, Puzzle Solving) achieve similar performance.
5. Key Takeaways
Deep networks trained on classification tasks can serve as powerful perceptual similarity metrics.
Even self-supervised networks show emergent perceptual similarity properties.
A simple linear weighting of deep features improves agreement with human perception.
Training on a perceptual dataset (like BAPPS) enhances the performance of these networks.
'''