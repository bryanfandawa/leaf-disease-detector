from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
import gradio as gr

try:
    import spaces
    gpu = spaces.GPU
except ImportError:
    gpu = lambda f: f
device = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = [
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

PRETTY = {
    "Pepper__bell___Bacterial_spot": "Pepper: Bacterial spot",
    "Pepper__bell___healthy": "Pepper: Healthy",
    "Potato___Early_blight": "Potato: Early blight",
    "Potato___Late_blight": "Potato: Late blight",
    "Potato___healthy": "Potato: Healthy",
    "Tomato_Bacterial_spot": "Tomato: Bacterial spot",
    "Tomato_Early_blight": "Tomato: Early blight",
    "Tomato_Late_blight": "Tomato: Late blight",
    "Tomato_Leaf_Mold": "Tomato: Leaf mold",
    "Tomato_Septoria_leaf_spot": "Tomato: Septoria leaf spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato: Spider mites",
    "Tomato__Target_Spot": "Tomato: Target spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato: Yellow leaf curl virus",
    "Tomato__Tomato_mosaic_virus": "Tomato: Mosaic virus",
    "Tomato_healthy": "Tomato: Healthy",
}

model = models.resnet18(weights = None)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model_path = Path(__file__).parent / "resnet_finetuned.pth"
model. load_state_dict(torch.load(model_path, map_location = "cpu"))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@gpu
def predict(image):
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim = 1)[0].cpu()

    results = {PRETTY[c]: float(p) for c, p in zip(CLASSES, probs)}
    top_class = CLASSES[probs.argmax()]
    top_prob = probs.max().item()

    if "healthy" in top_class and top_prob < 0.9:
        note = "Possibly diseased. Looks mostly healthy, but not confident enough to rule out disease. check the leaf closely"
    elif top_prob < 0.5:
        note = "Low confidence. Try a clear photo of a single leaf on a plain background"
    else:
        note = f"{PRETTY[top_class]} ({top_prob:.0%} confident)"
    return results, note

demo = gr.Interface(
    fn = predict,
    flagging_mode = "never",
    inputs = gr.Image(type = "pil", label = "Leaf photo"),
    outputs = [gr.Label(num_top_classes = 3, label = "Top predictions"), gr.Markdown()],
    title = "Leaf Disease Detector",
    description = "Upload a photo of a pepper, potato, or tomato leaf. "
                  "Trained on lab photos (99% lab accuracy), so it works best "
                  "on a single leaf with a plain background."
)

if __name__ == "__main__":
    demo.launch()