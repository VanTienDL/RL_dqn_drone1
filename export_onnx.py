import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from dqn import DQN

model = DQN()
model.load_state_dict(torch.load("model.pth"))
model.eval()

dummy_input = torch.randn(1, 6)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["state"],
    output_names=["q_values"],
    opset_version=11
)

print("Exported model.onnx")