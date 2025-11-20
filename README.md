# BERT Fine-Tuning Strategies for SST-2

본 프로젝트는 **BERT 기반 감정 분류(SST-2)** 문제를 대상으로, 다양한 **파인튜닝(Fine-Tuning)** 전략을 비교한 연구 프로젝트입니다.  
모든 실험은 동일한 **BERT-base-uncased 백본**, **동일한 데이터 분할 (Train/Validation/Test = 8:1:1)**, **동일한 평가 지표**를 사용하여 비교하였습니다.

------------------------------------------------------------

## 1. 비교한 Fine-Tuning 기법

아래는 본 프로젝트에서 실험한 파인튜닝 방식과 각 모델의 **학습 가능한 파라미터 수(Trainable Params)** 정리입니다.

| Model Name        | Backbone             | Fine-tuning Strategy        | Trainable Params (%) |
|-------------------|-----------------------|-----------------------------|-----------------------|
| Freeze FT         | bert-base-uncased     | Head-only (Classifier only) | 0.0014%               |
| Full Fine-tune     | bert-base-uncased     | Full Parameter Training     | 100%                  |
| Partial FT         | bert-base-uncased     | Top-4 Layers Only           | 25.897%               |
| BitFit             | bert-base-uncased     | Bias-only Training          | 0.094%                |
| LoRA               | bert-base-uncased     | Low-Rank Adaptation (r=8)   | 1.209956%             |



------------------------------------------------------------

## 2. 데이터셋 정보

• 데이터셋: SST-2 (Stanford Sentiment Treebank v2)  
• Train 데이터만 사용하여 **8:1:1 비율로** 재분할  
• 모든 수행 노트북에서 동일한 데이터 분할을 적용하여 재현성 확보


------------------------------------------------------------
## 3. 실험 방법 (Experiment Method)

본 프로젝트에서는 모든 파인튜닝 기법을 공정하게 비교하기 위해, 동일한 데이터 분할, 동일한 백본(BERT-base-uncased), 동일한 평가 방식(F1-score 중심)을 사용했습니다. 아래는 전체 실험 절차입니다.

---

### 1) 데이터 구성
- Hugging Face SST-2 데이터셋의 **train split만 로드**
- 8 : 1 : 1 비율로 **train / validation / test** 재구성
- 모든 실험에서 동일한 분할 사용

---

### 2) 하이퍼파라미터 탐색
각 파인튜닝 방식은 **Random Search**를 사용하여 최적 설정을 탐색했습니다.

탐색 범위:
- learning rate: 2e-5, 3e-5, 5e-5
- batch size: 16, 32, 64
- dropout: 0.1, 0.2
- epochs: 2, 3, 4

여러 trial 중 **validation Acc가 가장 높은 모델**을 최종 결과 비교에 사용했습니다.

---

### 3) 기법별 추가 설정

#### ▷ Freeze FT
- BERT 인코더 전체 freeze
- classifier(Linear)만 학습

#### ▷ Full Fine-Tuning
- 모든 레이어 파라미터 학습

#### ▷ Partial Fine-Tuning
-	BERT 인코더의 일부 레이어만 학습하도록 설정
-	k = {2, 4} 두 가지 설정 모두 실험
- 두 실험 중 k = 4가 더 높은 Validation 성능을 보여 최종 모델로 선택

#### ▷ BitFit
- 모든 레이어의 **bias 파라미터**만 학습

#### ▷ LoRA
- Attention의 Q, V projection에 저랭크 모듈 적용
- rank **r = 4, 8, 16** 각각 실험
- 그중 **r = 8 모델**을 최종 LoRA 결과로 사용

---

### 4) 평가 지표
모든 모델은 동일한 test split으로 평가했으며, 다음 지표를 계산했습니다.
- Accuracy
- Precision (macro)
- Recall (macro)
- **F1-score (macro)** → 최종 비교 기준으로 사용

---
## 4. 모델 아키텍처 


![BERT Architecture](https://github.com/user-attachments/assets/932d1c1b-cd05-4020-8731-4baddc865c20)

------------------------------------------------------------
## 5. 실험 결과
| Model Name        | Fine-tuning Strategy        | Trainable Params (%)       | F1 Score  |
|-------------------|-----------------------------|-----------------------------|----------|
| Freeze FT         | Head-only (Classifier only) | 0.0014%                     | 0.74  |
| Full Fine-tune     | Full Parameter Training     | 100%                        | 0.96   |
| Partial FT         | Top-4 Layers Only           | 25.897%                     | 0.94   |
| BitFit             | Bias-only Training          | 0.094%                      | 0.93   |
| LoRA              | Low-Rank Adaptation         | 1.209%                   | 0.92 |

![Test F1-scores](https://github.com/user-attachments/assets/c84fe4a4-8f3a-4974-9d02-236a033ddb4f)
## 6. 프로젝트 폴더 구조

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
|   ├── lora_test_outputs.csv
│   └── model_summary.csv
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

## 7. 폴더 및 파일 설명

📁 **models/**  
• GitHub에는 파일 용량 문제로 비워둡니다.  
• Hugging Face Hub에서 모델을 다운로드해 이 폴더에 배치하는 방식입니다.  
• 각 디렉토리는 해당 파인튜닝 방식의 모델을 저장하기 위한 폴더입니다.

📁 **results/**  
• 각 모델을 동일한 test split에서 평가한 결과(csv)를 저장합니다.  
• CSV 컬럼: `sentence, gold, pred`

📁 **notebooks/**  


Notebook | 설명
-------- | ----
freeze.ipynb | BERT Encoder 동결 + Linear Classifier만 학습
full_fine.ipynb | BERT 전체 파라미터 학습
partial_ft.ipynb | 마지막 K개의 encoder layer만 학습
bitfit.ipynb | Bias-only 튜닝 (BitFit)
lora.ipynb | LoRA 튜닝 (r = 4, 8, 16 등)
model_test.ipynb | 저장된 모든 모델을 동일한 test set에 대해 평가

------------------------------------------------------------

## 8. 실행 방법

### 1) 패키지 설치

```bash
pip install -r requirements.txt
```
### 2) 개별 실험 수행
예: Head-only Fine-Tuning → `notebooks/freeze.ipynb`

### 3) 저장된 모델 평가
`notebooks/model_test.ipynb` 실행

------------------------------------------------------------

## 9. Hugging Face Hub 모델 다운로드


📥 **예시: 모델 다운로드 방법**


```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "joononeyyy/bert-sst2-freeze"
)
tokenizer = AutoTokenizer.from_pretrained(
    "joononeyyy/bert-sst2-freeze"
)

print("Model and tokenizer loaded successfully!")
```

필요하신 경우 모델 이름만 바꿔서 사용할 수 있습니다:

- Freeze FT 모델  
  `joononeyyy/bert-sst2-freeze`

- Full Fine-tuning 모델  
  `joononeyyy/bert-sst2-full`

- Partial Fine-tuning 모델  
  `joononeyyy/bert-sst2-partial`

- BitFit 모델  
  `joononeyyy/bert-sst2-bitfit`

- LoRA 모델  
  `joononeyyy/bert-sst2-lora`





------------------------------------------------------------



