import torch
from transformers import AutoProcessor, AutoModelForPreTraining # Use the appropriate model class
from PIL import Image # Pillow is needed even if not directly used here, as processor might depend on it

# --- 1. モデルとプロセッサのロード (CPU指定) ---
model_id = "google/shieldgemma-2-4b-it"
device = "cpu"

try:
    processor = AutoProcessor.from_pretrained(model_id)
    # Replace AutoModelForPreTraining with the specific class if known
    model = AutoModelForPreTraining.from_pretrained(model_id)
    model.to(device)
    model.eval() # Set model to evaluation mode
    print(f"モデルとプロセッサをロードしました。実行デバイス: {device}")
except Exception as e:
    print(f"モデルまたはプロセッサのロード中にエラーが発生しました: {e}")
    exit()

# --- 2. 入力テキストと指示プロンプトの準備 ---
# Instruction Tuningを考慮したプロンプト形式の例
# 実際のモデルの期待する形式に合わせて調整が必要な場合があります
instruction = "以下のテキストの安全性を評価してください。\nテキスト: "
input_text = "これはテスト用の安全なテキストです。"
# input_text = "これは不適切な内容を示唆するテキストの例です。" # テスト用に切り替え可能

prompt = instruction + input_text

# --- 3. テキストの前処理 (CPUへ配置) ---
try:
    # プロセッサを使用してテキストをトークン化し、テンソルに変換
    # return_tensors="pt" はPyTorchテンソルを要求
    inputs = processor(text=prompt, return_tensors="pt").to(device)
    print("テキスト入力を前処理しました。")
except Exception as e:
    print(f"テキストの前処理中にエラーが発生しました: {e}")
    exit()

# --- 4. モデルによる推論 (CPUで実行) ---
print("推論を実行中 (CPU)...")
try:
    with torch.no_grad(): # 勾配計算を無効化し、メモリ消費と計算時間を削減
        outputs = model(**inputs)
    print("推論が完了しました。")

    # --- 5. 結果の解釈 ---
    # 出力形式はモデルに依存するため、以下は仮定に基づく例
    # 多くの場合、分類タスクでは logits が出力される
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
        print(f"出力 Logits (形状: {logits.shape}):\n{logits}")

        # Logitsから確率を計算 (例: Softmaxを使用)
        probabilities = torch.softmax(logits, dim=-1)
        print(f"出力確率:\n{probabilities}")

        # 最も確率の高いクラスIDを取得
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
        print(f"予測されたクラスID: {predicted_class_id}")

        # クラスIDをラベルにマッピング (モデルのドキュメントや設定に基づく必要がある)
        # これは仮の例です
        id2label = {0: "SAFE", 1: "UNSAFE"} # このマッピングはモデルに依存
        predicted_label = id2label.get(predicted_class_id, "UNKNOWN")
        print(f"予測されたラベル: {predicted_label}")

    else:
        # 他の出力形式の場合の処理 (モデルのドキュメントを参照)
        print("モデルの出力形式が想定外です。'logits' 属性が見つかりませんでした。")
        print(f"利用可能な出力キー: {outputs.keys()}")

except Exception as e:
    print(f"推論または結果解釈中にエラーが発生しました: {e}")