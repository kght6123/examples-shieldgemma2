from PIL import Image
import requests
from transformers import AutoProcessor, ShieldGemma2ForImageClassification

model_id = "google/shieldgemma-2-4b-it"
model = ShieldGemma2ForImageClassification.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

# 露出が多い温泉の写真、腕と顔が写っている
url = "https://user0514.cdnw.net/shared/img/thumb/SAYAPAKU4945_TP_V4.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=[image], return_tensors="pt").to(model.device)

output = model(**inputs)
print(output.probabilities)