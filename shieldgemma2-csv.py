import os
import argparse
import csv
from datetime import datetime, timezone, timedelta
import time
from PIL import Image
from transformers import AutoProcessor, ShieldGemma2ForImageClassification
import torch
import glob

# カスタムポリシーの定義
CUSTOM_POLICIES = {
    "explicit_genitalia": "The image contains clearly visible and explicit human sexual organs (e.g., penis, vagina, anus).",
}
# 評価するポリシーの順序を定義 https://github.com/huggingface/transformers/blob/3b3ebcec4077f124f2cd0ec3cd5d028dc352a3e5/src/transformers/models/shieldgemma2/convert_shieldgemma2_weights_orbax_to_hf.py#L63
POLICIES_TO_EVALUATE = ["dangerous", "sexual", "violence", "explicit_genitalia"]

# CSVヘッダー
CSV_HEADER = [
    "filename", "Model", "Prompt", "Created At (JST)", "Total Duration",
    "Load Duration", "Prompt Eval Duration", "Eval Duration", "Content",
    "Porno", "NotGenital", "Porno Score", "NotGenital Score",
    "Porno Reason Message", "NotGenital Reason Message", "Visible Genital Type"
]

# サポートする画像拡張子
SUPPORTED_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")

def main():
    parser = argparse.ArgumentParser(description="指定したフォルダの中にある画像の一覧をshieldgemma-2-4b-itで分類し、CSVに出力します。")
    parser.add_argument("image_folder", help="画像が格納されているフォルダのパス")
    parser.add_argument("--output_csv", default="shieldgemma2_classification_results.csv", help="出力CSVファイル名 (デフォルト: shieldgemma2_classification_results.csv)")
    args = parser.parse_args()

    if not os.path.isdir(args.image_folder):
        print(f"エラー: 指定されたフォルダが見つかりません: {args.image_folder}")
        return

    # モデルとプロセッサの読み込み
    model_id = "google/shieldgemma-2-4b-it"
    print(f"モデル '{model_id}' を読み込んでいます...")
    model_load_start_time = time.time()
    try:
        model = ShieldGemma2ForImageClassification.from_pretrained(model_id, device_map="auto")
        processor = AutoProcessor.from_pretrained(model_id)
        model.eval() # モデルを評価モードに設定
    except Exception as e:
        print(f"モデルまたはプロセッサの読み込み中にエラーが発生しました: {e}")
        return
    model_load_end_time = time.time()
    print(f"モデルの読み込み完了。所要時間: {model_load_end_time - model_load_start_time:.2f}秒")

    device = model.device
    print(f"モデルはデバイス '{device}' で実行されます。")

    image_files = []
    for ext_pattern in SUPPORTED_EXTENSIONS:
        # 大文字・小文字の拡張子を考慮し、再帰的に検索
        image_files.extend(glob.glob(os.path.join(args.image_folder, '**', ext_pattern.lower()), recursive=True))
        image_files.extend(glob.glob(os.path.join(args.image_folder, '**', ext_pattern.upper()), recursive=True))
    
    # 重複を除去
    image_files = sorted(list(set(image_files)))


    if not image_files:
        print(f"指定されたフォルダ '{args.image_folder}' にサポートされている画像ファイルが見つかりませんでした。")
        return

    print(f"{len(image_files)} 件の画像ファイルを処理します...")

    with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(CSV_HEADER)

        for image_path in image_files:
            print(f"処理中: {image_path}")
            file_eval_start_time = time.time()
            
            load_duration = 0.0
            prompt_eval_duration = 0.0
            eval_duration = 0.0

            try:
                # 1. 画像の読み込み
                image_load_start = time.time()
                image = Image.open(image_path).convert("RGB") # RGBに変換してチャンネル数を統一
                image_load_end = time.time()
                load_duration = image_load_end - image_load_start

                # 2. 前処理
                preprocess_start = time.time()
                inputs = processor(
                    images=[image],
                    custom_policies=CUSTOM_POLICIES,
                    policies=POLICIES_TO_EVALUATE, # 定義した順序でポリシーを指定
                    return_tensors="pt",
                ).to(device)
                preprocess_end = time.time()
                prompt_eval_duration = preprocess_end - preprocess_start

                # 3. 推論の実行
                inference_start = time.time()
                with torch.no_grad(): # 勾配計算を無効化してメモリ効率を上げる
                    output = model(**inputs)
                inference_end = time.time()
                eval_duration = inference_end - inference_start
                
                # probabilitiesを出力する
                print(f"Probabilities for {image_path}: {output.probabilities.tolist()}")
                
                # POLICIES_TO_EVALUATE の順序に基づいて確率を取得
                porno_score_raw = output.probabilities[POLICIES_TO_EVALUATE.index("sexual")]
                genitalia_covered_score_raw = output.probabilities[POLICIES_TO_EVALUATE.index("explicit_genitalia")]

                porno_result = porno_score_raw[0] > 0.5
                # CSVのNotGenitalは「覆われているか、性器がない」ポリシーの結果
                not_genital_result = genitalia_covered_score_raw[0] > 0.5 

                porno_score_percent1 = int(porno_score_raw[0] * 100)
                porno_score_percent2 = int(porno_score_raw[1] * 100)
                not_genital_score_percent1 = int(genitalia_covered_score_raw[0] * 100)
                not_genital_score_percent2 = int(genitalia_covered_score_raw[1] * 100)

                created_at_jst = datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S')
                
                file_eval_end_time = time.time()
                total_duration = file_eval_end_time - file_eval_start_time

                # Content列の文字列を生成
                content_details = []
                content_details.append(f"Porno=ResultDetail(value={porno_result}, score={porno_score_percent1}, {porno_score_percent2}, reasonMessage=sexual')")
                content_details.append(f"NotGenital=ResultDetail(value={not_genital_result}, score={not_genital_score_percent1}, {not_genital_score_percent2}, reasonMessage='{CUSTOM_POLICIES['explicit_genitalia']}')")
                content_str = " ".join(content_details)

                row = [
                    image_path,
                    model_id,
                    CUSTOM_POLICIES['explicit_genitalia'],
                    created_at_jst,
                    f"{total_duration:.6f}",
                    f"{load_duration:.6f}",
                    f"{prompt_eval_duration:.6f}",
                    f"{eval_duration:.6f}",
                    content_str,
                    porno_result,
                    not_genital_result,
                    porno_score_percent1,
                    not_genital_score_percent1,
                    porno_score_percent2, # Porno Reason Message
										not_genital_score_percent2, # NotGenital Reason Message
                    "" # Visible Genital Type
                ]
                csv_writer.writerow(row)

            except Exception as e:
                print(f"ファイル '{image_path}' の処理中にエラーが発生しました: {e}")
                error_created_at_jst = datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S')
                error_row = [
                    image_path, model_id, "", error_created_at_jst, 
                    f"{(time.time() - file_eval_start_time):.6f}" if file_eval_start_time else "0.000000",
                    f"{load_duration:.6f}", f"{prompt_eval_duration:.6f}", f"{eval_duration:.6f}",
                    f"Error: {str(e)}", 
                    "", "", "", "", "", "", ""
                ]
                csv_writer.writerow(error_row)
                continue
        
    print(f"処理が完了しました。結果は '{args.output_csv}' に保存されました。")

if __name__ == "__main__":
    main()