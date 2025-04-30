from PIL import Image
import requests
from transformers import AutoProcessor, ShieldGemma2ForImageClassification

model_id = "google/shieldgemma-2-4b-it"
model = ShieldGemma2ForImageClassification.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

# ラボ子ちゃんの露出があるイラスト
url = "20_thinking.png"
image = Image.open(url)

inputs = processor(images=[image], return_tensors="pt").to(model.device)

output = model(**inputs)
print(output.probabilities)