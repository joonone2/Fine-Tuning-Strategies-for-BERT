# BERT Fine-Tuning Strategies for SST-2

본 프로젝트는 **BERT 기반 감정 분류(SST-2)** 문제를 대상으로, 다양한 **파인튜닝(Fine-Tuning)** 전략을 비교한 연구 프로젝트입니다.  
모든 실험은 동일한 **BERT-base-uncased 백본**, **동일한 데이터 분할 (Train/Validation/Test = 8:1:1)**, **동일한 평가 지표**를 사용하여 공정하게 비교하였습니다.

------------------------------------------------------------

## 1. 비교한 Fine-Tuning 기법

아래는 본 프로젝트에서 실험한 파인튜닝 방식과 각 모델의 **학습 가능한 파라미터 수(Trainable Params)** 정리입니다.

Model ID | Backbone | Tuning Strategy | Trainable Params
-------- | -------- | --------------- | ------------------------------
M1 | bert-base-uncased | Head-only (CLS Linear Only) | 1,538 / 109,483,778 (0.0014%)
M2 | bert-base-uncased | Full Fine-tuning | 109,483,778 / 109,483,778 (100%)
M3 | bert-base-uncased | Partial FT (Top-4 layers) | 28,353,026 / 109,483,778 (25.897%)
M4 | bert-base-uncased | BitFit (Bias-only Tuning) | 102,914 / 109,483,778 (0.094%)
M5 | bert-base-uncased | LoRA (r = 4 / 8 / 16) | 약 0.1% ~ 1%



------------------------------------------------------------

## 2. 데이터셋 정보

• 데이터셋: SST-2 (Stanford Sentiment Treebank v2)  
• Train 데이터만 사용하여 **8:1:1 비율로** 재분할  
• 모든 수행 노트북에서 동일한 데이터 분할을 적용하여 재현성 확보


------------------------------------------------------------

## 3. 프로젝트 폴더 구조

## 📁 프로젝트 폴더 구조

```
project/
│
├── models/                     # 모델 저장 폴더
│   ├── freeze_model/           # Head-only fine-tuning 결과 모델
│   ├── full_fine_model/        # Full fine-tuning 결과 모델
│   ├── partial_ft_model/       # Partial (Top-k layers) fine-tuning 모델
│   ├── bitfit_model/           # Bias-only (BitFit) fine-tuning 모델
│   └── lora_model/             # LoRA fine-tuning 모델
│
├── results/                    # 테스트셋에 대한 예측 결과 CSV 파일
│   ├── freeze_test_outputs.csv
│   ├── full_test_outputs.csv
│   ├── partial_test_outputs.csv
│   ├── bitfit_test_outputs.csv
│   └── lora_test_outputs.csv
│
├── data/                       # SST-2 원본 데이터 저장 폴더
│   └── sst2_raw/
│
├── notebooks/                  # 개별 실험을 실행 Notebook 파일
│   ├── freeze.ipynb            # Freeze(Head-only) 실험
│   ├── full_fine.ipynb         # Full fine-tuning 실험
│   ├── partial_ft.ipynb        # Partial fine-tuning 실험
│   ├── bitfit.ipynb            # BitFit 실험
│   ├── lora.ipynb              # LoRA 실험
│   └── model_test.ipynb        # 모든 모델을 동일한 testset으로 평가
│
├── README.md                   
└── requirements.txt           
```

------------------------------------------------------------

## 4. 폴더 및 파일 설명

📁 **models/**  
• GitHub에는 파일 용량 문제로 비워둡니다.  
• Hugging Face Hub에서 모델을 다운로드해 이 폴더에 배치하는 방식입니다.  
• 각 디렉토리는 해당 파인튜닝 방식의 모델을 저장하기 위한 폴더입니다.

📁 **results/**  
• 각 모델을 동일한 test split에서 평가한 결과(csv)를 저장합니다.  
• CSV 컬럼: `sentence, gold, pred`

📁 **notebooks/  
각 실험은 독립적인 Jupyter Notebook으로 구성되어 있습니다.

Notebook | 설명
-------- | ----
freeze.ipynb | BERT Encoder 동결 + Linear Classifier만 학습
full_fine.ipynb | BERT 전체 파라미터 학습
partial_ft.ipynb | 마지막 K개의 encoder layer만 학습
bitfit.ipynb | Bias-only 튜닝 (BitFit)
lora.ipynb | LoRA 튜닝 (r = 4, 8, 16 등)
model_test.ipynb | 저장된 모든 모델을 동일한 test set에 대해 평가

------------------------------------------------------------

## 5. 실행 방법

### 1) 패키지 설치

```bash
pip install -r requirements.txt
```
### 2) 개별 실험 수행
예: Head-only Fine-Tuning → `notebooks/freeze.ipynb`

### 3) 저장된 모델 평가
`notebooks/model_test.ipynb` 실행

------------------------------------------------------------

## 6. Hugging Face Hub 모델 다운로드

각 파인튜닝 방식의 모델은 Hugging Face Hub에 업로드되어 있으며,  
다운로드 후 `models/` 폴더 내부에 배치하여 사용합니다.

freeze_model:     https://huggingface.co/joononeyyy/freeze-sst2  
full_fine_model:  https://huggingface.co/joononeyyy/full-sst2  
partial_ft_model: https://huggingface.co/joononeyyy/partial-sst2  
bitfit_model:     https://huggingface.co/joononeyyy/bitfit-sst2  
lora_model:       https://huggingface.co/joononeyyy/lora-sst2  

------------------------------------------------------------

## 7. 평가 지표

모든 모델은 동일한 평가 지표로 성능을 비교하였습니다.

• Accuracy  
• Precision (macro)  
• Recall (macro)  
• F1-score (macro)

------------------------------------------------------------

## 8. 성능 요약
| Model Name        | Fine-tuning Strategy        | Trainable Params (%)       | Test F1  |
|-------------------|-----------------------------|-----------------------------|----------|
| Freeze FT         | Head-only (Classifier only) | 0.0014%                     | 0.74  |
| Full Fine-tune     | Full Parameter Training     | 100%                        | 0.96   |
| Partial FT         | Top-4 Layers Only           | 25.897%                     | 0.94   |
| BitFit             | Bias-only Training          | 0.094%                      | 0.93   |
