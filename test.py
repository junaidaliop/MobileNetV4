import torch
import torch.nn as nn
from MobileNetV4 import build_mobilenet
from torchsummary import summary
import sys
import os

# BATCH_SIZE = 2 FOR SMALL.
BATCH_SIZE = 128

class MobileNetV4WithClassifier(nn.Module):
    def __init__(self, model_name, num_classes=100, dropout_rate=0.2, input_size=(3, 224, 224)):
        super(MobileNetV4WithClassifier, self).__init__()
        self.features = build_mobilenet(model_name, input_specs=input_size)
        
        with torch.no_grad():
            dummy_input = torch.randn(BATCH_SIZE, *input_size)
            features = self.features(dummy_input)
            num_features = features.shape[1]
        
        self.classifier = nn.Sequential(
            nn.Conv2d(1280, num_classes, kernel_size=1),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def print_model_structure(model_name, input_shape=(3, 224, 224)):
    original_stdout = sys.stdout
    filename = os.path.join('logs', f"{model_name}_architecture.txt")
    
    with open(filename, 'w') as f:
        sys.stdout = f
        
        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"{'='*50}")
        
        try:
            model = MobileNetV4WithClassifier(model_name, num_classes=100, input_size=input_shape)
            model.eval()
            
            summary(model, input_shape, device="cpu", batch_size=BATCH_SIZE)
            
            print("\nDetailed layer shapes:")
            dummy_input = torch.randn(BATCH_SIZE, *input_shape)
            x = dummy_input
            for name, layer in model.named_modules():
                if not list(layer.children()):
                    try:
                        with torch.no_grad():
                            x = layer(x)
                        print(f"{name}: {x.shape}")
                    except Exception as e:
                        print(f"Error in layer {name}: {str(e)}")

            print(f"\nFinal output shape: {x.shape}")
        except Exception as e:
            print(f"Error creating or analyzing model: {str(e)}")
        
        print(f"{'='*50}\n")
        
        sys.stdout = original_stdout

models = [
    ('MobileNetV4ConvSmall', (3, 224, 224)),
    ('MobileNetV4ConvMedium', (3, 256, 256)),
    ('MobileNetV4ConvLarge', (3, 384, 384)),
    ('MobileNetV4HybridMedium', (3, 256, 256)),
    ('MobileNetV4HybridLarge', (3, 384, 384))
]

os.makedirs('logs', exist_ok=True)

for model_name, input_shape in models:
    print(f"Processing {model_name}")
    print_model_structure(model_name, input_shape)
    print(f"Finished processing {model_name}")
