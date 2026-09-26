import torch
import torch.nn as nn
import torchvision 


class VGG16(nn.Module):
    def __init__(self, weights_path=None):
        super(VGG16, self).__init__()
        if weights_path:
            vgg16 = torchvision.models.vgg16(weights=None)
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            vgg16.load_state_dict(state, strict=True)
        else:
            # Keep the legacy pretrained behaviour for existing F/G paths.
            vgg16 = torchvision.models.vgg16(pretrained=True)

        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        for i in range(3):
            for param in getattr(self, f'enc_{i+1:d}').parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, f'enc_{i+1:d}')
            results.append(func(results[-1]))
        return results[1:]


class ContentPerceptualLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.VGG = VGG16()

    def calculate_loss(self, generated_images, target_images, device):
        self.VGG = self.VGG.to(device)

        generated_features = self.VGG(generated_images)
        target_features = self.VGG(target_images)

        perceptual_loss = 0
        perceptual_loss += torch.mean((target_features[0] - generated_features[0]) ** 2)
        perceptual_loss += torch.mean((target_features[1] - generated_features[1]) ** 2)
        perceptual_loss += torch.mean((target_features[2] - generated_features[2]) ** 2)
        perceptual_loss /= 3
        return perceptual_loss
