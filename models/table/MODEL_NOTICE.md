# Table numeric recognition model

- Model: en_PP-OCRv4_rec_mobile.onnx (PaddleOCR English recognition, RapidAI ONNX distribution)
- Download: https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv4/rec/en_PP-OCRv4_rec_mobile.onnx
- SHA256: e8770c967605983d1570cdf5352041dfb68fa0c21664f49f47b155abd3e0e318
- Official registry: https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/default_models.yaml
- Upstream license: Apache-2.0; https://github.com/RapidAI/RapidOCR/blob/main/LICENSE
- Copyright (c) 2021 RapidOCR Authors. All rights reserved.
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR

Recognition runs entirely on the local OCR runtime. This model is not downloaded during normal OCR, and images are not sent to an external service. The app's table worker implements aspect-preserving input resizing and CTC decoding; it does not copy RapidOCR package source code.
