import torch
from model.base_model import load_base_model

def fake_quantize_tensor(tensor: torch.Tensor, num_bits: int):
    num_levels = 2 ** num_bits
    min_val = tensor.min()
    max_val = tensor.max()

    # if the tensor has no range (all values identical), skip quantization entirely
    if (max_val - min_val).item() < 1e-8:
        return tensor

    scale = (max_val - min_val) / (num_levels - 1)
    quantized = torch.round((tensor - min_val) / scale)
    dequantized = quantized * scale + min_val

    return dequantized

def quantize_model(model, num_bits: int):
    """Apply fake quantization to all weight tensors in the model."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.data = fake_quantize_tensor(param.data, num_bits)
    return model

if __name__ == "__main__":
    model_8bit = load_base_model()
    model_8bit = quantize_model(model_8bit, num_bits=8)
    print("8-bit quantization applied.")

    model_4bit = load_base_model()
    model_4bit = quantize_model(model_4bit, num_bits=4)
    print("4-bit quantization applied.")