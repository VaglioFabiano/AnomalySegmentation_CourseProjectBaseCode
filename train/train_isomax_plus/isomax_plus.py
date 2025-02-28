import torch.nn as nn
import torch.nn.functional as F
import torch


class IsoMaxPlusLossFirstPart(nn.Module):
    """This part replaces the model classifier output layer nn.Linear()"""
    def __init__(self, num_features, num_classes, temperature=1.0):
        super(IsoMaxPlusLossFirstPart, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.temperature = temperature        
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, num_features))
        self.distance_scale = nn.Parameter(torch.Tensor(1)) 
        nn.init.normal_(self.prototypes, mean=0.0, std=1.0)
        nn.init.constant_(self.distance_scale, 1.0)

    def forward(self, features):
        batch,channels,height,width = features.shape
        features = features.permute(0,2,3,1).contiguous()
        features = features.view(-1,channels)
        with torch.cuda.amp.autocast(): 
            distances = torch.abs(self.distance_scale) * torch.cdist(F.normalize(features), F.normalize(self.prototypes), p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
            logits = -distances
        # The temperature may be calibrated after training to improve uncertainty estimation.
        return logits / self.temperature


class IsoMaxPlusLossSecondPart(nn.Module):
    """This part replaces the nn.CrossEntropyLoss()"""
    def __init__(self, entropic_scale=10.0):
        super(IsoMaxPlusLossSecondPart, self).__init__()
        self.entropic_scale = entropic_scale

    def forward(self, logits, targets, debug=False):
        with torch.cuda.amp.autocast():
            distances = -logits
            probabilities_for_training = nn.Softmax(dim=1)(-self.entropic_scale * distances).view(-1,logits.size(-1))
            targets = targets.view(-1)
            probabilities_at_targets = probabilities_for_training[range(distances.view(-1,logits.size(-1)).size(0)), targets]
            loss = -torch.log(probabilities_at_targets).mean()
            del distances
        
        return loss
        