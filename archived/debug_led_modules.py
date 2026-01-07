
from transformers import LEDForConditionalGeneration
import torch

def inspect_model():
    print("Loading LED model...")
    model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")
    
    print("\nListing module class names:")
    attention_classes = set()
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name == "LEDEncoderSelfAttention":
            print(f"Inspecting {cls_name}: {name}")
            print("  Child modules:")
            for child_name, child in module.named_children():
                print(f"    - {child_name}: {child.__class__.__name__}")
            break
            
        if cls_name == "LEDDecoderAttention" and hasattr(module, 'q_proj'):
             pass # Already confirmed
            
    print("\nUnique Attention Classes found:", attention_classes)

if __name__ == "__main__":
    inspect_model()
