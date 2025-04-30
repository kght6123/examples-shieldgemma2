# shieldgemma-2 のサンプルコード

検証内容は Tech Talk #4 を確認してください。

## 前提条件

- huggingface-cli login でログインしてください


## 環境の作り方

このサンプルを構築した手順を参考に記載します

```sh
cd ./examples-shieldgemma2
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
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

pip freeze > requirements.txt # 依存関係の出力
deactivate
```

## 参考資料

- [Hugging Face Transformers Installation Guide](https://huggingface.co/docs/transformers/ja/installation)
- https://huggingface.co/docs/transformers/main/model_doc/shieldgemma2
