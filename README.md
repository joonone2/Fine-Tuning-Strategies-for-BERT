아래는 바로 GitHub README.md에 붙여넣을 수 있는 최종 버전이다.
Markdown 형식을 깔끔하게 정리했고, 중요한 부분은 적절히 굵게 강조해두었다.

⸻

Fine-Tuning Strategies for BERT on SST-2

본 프로젝트는 **BERT 기반 문장 감정 분류(SST-2)**를 대상으로 여러 파인튜닝(Fine-Tuning) 방식을 비교하고, 각 기법의 효율성·성능·학습 파라미터 수를 실험적으로 분석하는 것을 목표로 한다.

모든 실험은 동일한 데이터셋(SST-2), 동일한 데이터 분할(Train 8:1:1 → Train/Validation/Test), 동일한 평가 지표(Accuracy, Precision, Recall, F1-score)를 사용한다.
또한 모든 실험은 동일한 bert-base-uncased 백본을 사용한다.

⸻

1. 비교한 Fine-Tuning 기법

아래는 본 프로젝트에서 비교한 5가지 파인튜닝 전략과 학습되는 파라미터 수이다.

Model ID	Backbone	Tuning Strategy	Trainable Params
M1	bert-base-uncased	Head-only (CLS Linear)	1,538 / 109,483,778 (0.0014%)
M2	bert-base-uncased	Full Fine-tuning	109,483,778 / 109,483,778 (100%)
M3	bert-base-uncased	Partial FT (Top-4 layers)	28,353,026 / 109,483,778 (25.897%)
M4	bert-base-uncased	BitFit (Bias-only)	102,914 / 109,483,778 (0.094%)
M5	bert-base-uncased	LoRA (r = 4, 8, 16)	약 0.1% ~ 1% 수준

각 방식은 “성능 ↔ 비용” 관점에서 우열이 나뉘며, 학습 유지 비용과 효율성을 정량적으로 비교할 수 있다.

⸻

2. 데이터셋
	•	데이터: SST-2 (Stanford Sentiment Treebank v2)
	•	로딩: load_dataset("glue", "sst2")
	•	분할: Train split을 8:1:1 비율로 재분할하여 Train/Validation/Test 생성
	•	모든 실험 노트북에 동일한 데이터 로딩/분할 코드 포함

별도의 데이터 파일을 저장하지 않으며, Hugging Face 데이터 캐시를 자동 사용한다.

⸻

3. 폴더 구조

본 프로젝트는 아래와 같은 구조로 구성되어 있다.

project/
│
├── models/                             # 로컬 테스트용 모델 (GitHub에는 빈 폴더)
│   ├── freeze_model/
│   ├── full_fine_model/
│   ├── partial_ft_model/
│   ├── bitfit_model/
│   └── lora_model/
│
├── results/                            # 테스트셋 예측 결과 (CSV)
│   ├── freeze_test_outputs.csv
│   ├── full_test_outputs.csv
│   ├── partial_test_outputs.csv
│   ├── bitfit_test_outputs.csv
│   └── lora_test_outputs.csv
│
├── data/                               # (선택) 원본 데이터 저장용
│   └── sst2_raw/
│
├── notebooks/
│   ├── freeze.ipynb                    # Head-only Fine-tuning
│   ├── full_fine.ipynb                 # Full Fine-tuning
│   ├── partial_ft.ipynb                # Last-k Layers Fine-tuning
│   ├── bitfit.ipynb                    # BitFit (bias-only)
│   ├── lora.ipynb                      # LoRA
│   ├── model_test.ipynb                # 여러 모델 test 평가
│   └── (각 노트북은 load_dataset + tokenization + training 포함)
│
├── README.md
└── requirements.txt


⸻

4. 각 폴더 및 파일 설명

models/
	•	GitHub에는 용량 문제로 비어 있으며,
Hugging Face Hub에서 다운로드한 모델을 이 폴더에 배치하여 사용한다.
	•	각 폴더는 다음 모델을 저장하기 위해 존재한다.
	•	freeze_model/
	•	full_fine_model/
	•	partial_ft_model/
	•	bitfit_model/
	•	lora_model/

⸻

results/
	•	각 모델을 동일한 test set으로 평가한 결과를 보관
	•	CSV 구성:
sentence, gold, pred

⸻

notebooks/

각 노트북은 독립적으로 실행 가능하도록 구성되어 있다.

Notebook	설명
freeze.ipynb	BERT 전체 freeze → 분류기(head)만 학습
full_fine.ipynb	BERT full fine-tuning
partial_ft.ipynb	BERT top-k layers fine-tuning
bitfit.ipynb	Bias-only fine-tuning
lora.ipynb	LoRA adapter fine-tuning
model_test.ipynb	모든 모델을 불러와 같은 test 데이터로 평가


⸻

5. 실행 방법

1) 패키지 설치

pip install -r requirements.txt

2) 각 실험 수행

예:

notebooks/freeze.ipynb

3) 저장된 모델로 테스트 실행

notebooks/model_test.ipynb


⸻

6. 모델 다운로드 (Hugging Face Hub)

아래 모델들은 모두 Hugging Face Hub에 업로드되어 있다.
원하는 모델을 다운로드하여 /models/ 내부에 넣어 사용한다.

예시 링크(사용자 계정에 맞게 수정):

freeze_model:      https://huggingface.co/<username>/freeze-sst2
full_fine_model:   https://huggingface.co/<username>/full-sst2
partial_ft_model:  https://huggingface.co/<username>/partial-sst2
bitfit_model:      https://huggingface.co/<username>/bitfit-sst2
lora_model:        https://huggingface.co/<username>/lora-sst2


⸻

7. 평가 지표

모든 실험은 아래 네 가지 지표로 평가했다.
	•	Accuracy
	•	Precision (macro)
	•	Recall (macro)
	•	F1-score (macro)

⸻

원하면 아래 항목도 추가해줄 수 있다:
	•	최종 성능 비교표 (M1~M5)
	•	그래프(학습 곡선, 파라미터 대비 성능)
	•	결론 및 분석 섹션
	•	추가 개선 방법

말해줘, 바로 추가해줄게!
