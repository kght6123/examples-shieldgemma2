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
from PIL import Image
import argparse
import requests
import os
from transformers import AutoProcessor, ShieldGemma2ForImageClassification
from urllib.parse import urlparse

def is_url(string):
    """文字列がURLかどうかを判定する"""
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except:
        return False

def load_image(image_path):
    """URLまたはファイルパスから画像を読み込む"""
    if is_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)
    else:
        raise ValueError("有効なURLまたはファイルパスを指定してください。")

def main():
    parser = argparse.ArgumentParser(description='画像分類を実行します')
    parser.add_argument('image_path', help='画像のURLまたはファイルパス')
    args = parser.parse_args()

    # モデルとプロセッサの読み込み
    model_id = "google/shieldgemma-2-4b-it"
    model = ShieldGemma2ForImageClassification.from_pretrained(model_id, device_map="auto")
    processor = AutoProcessor.from_pretrained(model_id)

    try:
        # 画像の読み込み
        image = load_image(args.image_path)
        
        # 推論の実行
        inputs = processor(images=[image], return_tensors="pt").to(model.device)
        output = model(**inputs)
        print(f"入力画像: {args.image_path}")
        print("分類結果:")
        print(output.probabilities)
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()
