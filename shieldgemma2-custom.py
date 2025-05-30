from PIL import Image
import requests
from transformers import AutoProcessor, ShieldGemma2ForImageClassification

model_id = "google/shieldgemma-2-4b-it"
model = ShieldGemma2ForImageClassification.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
image = Image.open(requests.get(url, stream=True).raw)

custom_policies = {
    "key_a": "descrition_a",
    "key_b": "descrition_b",
}

inputs = processor(
    images=[image],
    custom_policies=custom_policies,
    policies=["dangerous", "key_a", "key_b"],
    return_tensors="pt",
).to(model.device)

output = model(**inputs)
print(output.probabilities)