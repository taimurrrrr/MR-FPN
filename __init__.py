from .resnet import resnet18, resnet50, resnet101
from .nlnet import nlnet18, nlnet34, nlnet50
from .builder import build_backbone
from .CoT_ResNet import cotnet50
from .GCnet import GC_resnet50
from .SA_Net import sa_resnet50
__all__ = ['resnet18', 'resnet50', 'resnet101', 'cotnet50',  'nlnet18', 'nlnet34',  'nlnet50', 'GC_resnet50', 'sa_resnet50' ]
