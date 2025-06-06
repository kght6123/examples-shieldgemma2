# shieldgemma-2 のサンプルコード

検証内容は Tech Talk #4 を確認してください。

## 前提条件

- huggingface-cli login でログインしてください
- https://huggingface.co/google/shieldgemma-2-4b-it でモデルへのアクセス許可を取得してください


## 環境の作り方

このサンプルを構築した手順を参考に記載します

```sh
cd ./examples-shieldgemma2
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate or .venv\Scripts\activate.bat
pip install "transformers[torch]" Pillow #  Transformers と PyTorch、Pillow のインストール
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))" # 確認

python shieldgemma2-buildin.py
nohup python shieldgemma2-buildin.py > python.log 2>&1 &

# tensor([[5.2850e-11, 1.0000e+00],
#        [6.7467e-09, 1.0000e+00],
#        [3.1827e-17, 1.0000e+00]], grad_fn=<SoftmaxBackward0>)

# user	5m50.343s

python shieldgemma2-custom.py

# tensor([[5.2849e-11, 1.0000e+00],
#        [3.0857e-09, 1.0000e+00],
#        [3.7217e-09, 1.0000e+00]], grad_fn=<SoftmaxBackward0>)

# user	5m50.628s

# M3 Pro 32GBでは足りない
# RuntimeError: MPS backend out of memory (MPS allocated: 42.33 GB, other allocations: 3.00 GB, max allowed: 45.90 GB). Tried to allocate 3.00 GB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).

# 同じ画像なら何度実行しても同じ結果になりました

# paku1

tensor([[1.3077e-08, 1.0000e+00],
        [1.1047e-02, 9.8895e-01],
        [1.0645e-11, 1.0000e+00]], grad_fn=<SoftmaxBackward0>)

# paku2

tensor([[1.0985e-08, 1.0000e+00],
        [3.1323e-04, 9.9969e-01],
        [1.8727e-12, 1.0000e+00]], grad_fn=<SoftmaxBackward0>)

# paku3

tensor([[4.1718e-07, 1.0000e+00],
        [1.3430e-03, 9.9866e-01],
        [1.1559e-09, 1.0000e+00]], grad_fn=<SoftmaxBackward0>)

# lab1

tensor([[4.2510e-07, 1.0000e+00],
        [6.5711e-02, 9.3429e-01],
        [3.5839e-07, 1.0000e+00]], grad_fn=<SoftmaxBackward0>)

# lab2

tensor([[4.4427e-08, 1.0000e+00],
        [2.3781e-03, 9.9762e-01],
        [1.1112e-07, 1.0000e+00]], grad_fn=<SoftmaxBackward0>)

# PNG画像はJPEGに変換して実行する必要がありそう！！

python shieldgemma2.py ./huusenPAKE5506_TP_V4.jpg

pip freeze > requirements.txt # 依存関係の出力
deactivate
```

## 結果

```
入力画像: ./huusenPAKE5506_TP_V4.jpg
分類結果:
tensor([[9.9653e-10, 1.0000e+00],
        [4.3994e-07, 1.0000e+00],
        [3.2287e-11, 1.0000e+00]], grad_fn=<SoftmaxBackward0>)

入力画像: ./IMG_2568.jpg
分類結果:
tensor([[9.8415e-11, 1.0000e+00],
        [8.7994e-11, 1.0000e+00],
        [2.2427e-16, 1.0000e+00]], grad_fn=<SoftmaxBackward0>)


入力画像: ./yat4M3A7406_TP_V4.jpg
分類結果:
tensor([[8.3028e-08, 1.0000e+00],
        [1.1623e-05, 9.9999e-01],
        [2.5790e-10, 1.0000e+00]], grad_fn=<SoftmaxBackward0>)

入力画像: ./zubotty-hasedsc_2187_TP_V4.jpg
分類結果:
tensor([[8.5990e-10, 1.0000e+00],
        [3.9293e-07, 1.0000e+00],
        [1.0889e-12, 1.0000e+00]], grad_fn=<SoftmaxBackward0>)

入力画像: ./zubotty-hasedsc_2187_TP_V4.jpg
分類結果:
tensor([[8.5990e-10, 1.0000e+00],
        [3.9293e-07, 1.0000e+00],
        [1.0889e-12, 1.0000e+00]], grad_fn=<SoftmaxBackward0>)

```

## CSV

```sh
python -u shieldgemma2-csv.py ./images --output_csv rating-20250528.csv
# バックググラウンドで実行する場合はnohupを使うと良いでしょう
nice -n -20 nohup python -u shieldgemma2-csv.py ./images --output_csv rating-20250528.csv > shieldgemma2-csv.log 2>&1 &
tail -f shieldgemma2-csv.log
```

## TODO

- [x] 他の危険な画像をHuggingFaceから指定して実行する
- [ ] 危険なテキストを指定して実行する
- [ ] サンプルコードの処理内容を整理する
- [ ] 資料を作る

## 参考資料

- [Hugging Face Transformers Installation Guide](https://huggingface.co/docs/transformers/ja/installation)
- https://huggingface.co/docs/transformers/main/model_doc/shieldgemma2
- https://huggingface.co/google/shieldgemma-2b
- https://huggingface.co/google/shieldgemma-2-4b-it
- https://ai.google.dev/responsible/docs/safeguards/shieldgemma?hl=ja
- https://colab.research.google.com/github/google/generative-ai-docs/blob/main/site/en/responsible/docs/safeguards/shieldgemma_on_huggingface.ipynb?hl=ja#scrollTo=av03uUlhHeYq
- https://colab.research.google.com/github/google/generative-ai-docs/blob/main/site/en/responsible/docs/safeguards/shieldgemma2_on_huggingface.ipynb?hl=ja#scrollTo=2b40722aa1a9
- https://github.com/google/generative-ai-docs/blob/main/site/en/responsible/docs/safeguards/shieldgemma2_on_huggingface.ipynb
