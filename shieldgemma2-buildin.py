from PIL import Image
"""
このスクリプトは、Hugging Face Transformersライブラリの
ShieldGemma2ForImageClassificationモデルを使用して画像を分類する方法を示します。

次の手順を実行します。
1. 事前学習済みの 'google/shieldgemma-2-4b-it' モデルとその関連プロセッサをロードします。
2. 指定されたURLから画像（蜂）をダウンロードします。
3. ロードされたプロセッサを使用して画像を前処理します。
4. 前処理された画像に対してShieldGemma2モデルを使用して推論を実行します。
5. 結果の分類確率をコンソールに出力します。
"""
import requests
from transformers import AutoProcessor, ShieldGemma2ForImageClassification

model_id = "google/shieldgemma-2-4b-it"
model = ShieldGemma2ForImageClassification.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=[image], return_tensors="pt").to(model.device)

output = model(**inputs)
print(output.probabilities)