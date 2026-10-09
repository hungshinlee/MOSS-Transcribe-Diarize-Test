# MOSS-Transcribe-Diarize 實驗改進與修復專案 (Mac Apple Silicon 支援)

本專案自 `/Users/winston/Documents/moss_repeat_repro` 派生，專注於：
1. **重現與定位**：Fine-tune 檢查點（`checkpoint-8000-deploy`）出現的「重複輸出退化（Repeat Degeneration）」問題。
2. **推論防護**：評測與整合推論端防重複機制（`repetition_penalty`、`no_repeat_ngram_size`）。
3. **微調改進**：探索更穩健的微調方案（凍結 Whisper Encoder、LoRA 輕量微調、噪聲與真實語料擴增）。

---

## 目錄結構

本資料夾為完全獨立自包含的專案，已包含音訊資料集與模型權重：

```
MOSS-Transcribe-Diarize-Test/
├── model/
│   ├── checkpoint-8000-deploy/   # 會重複退化的微調模型 (約 3.6GB)
│   └── pretrain-baseline/        # 原始預訓練模型（對照組，約 3.6GB）
├── data/
│   ├── dev.jsonl / dev_audio/    # 10 句真人驗證集
│   ├── hotword_benchmark/        # 66 句 Benchmark 評測集與音檔
│   └── train_full.jsonl / ...    # 訓練資料與未解壓完整音檔
├── finetune_official/            # 官方模型定義與架構程式碼 (trust_remote_code)
├── scripts/                      # 評測與微調腳本 (已適配 Mac MPS 加速)
│   ├── eval_hotword_testset.py   # 支援 --device mps 與 --repetition_penalty
│   ├── eval_dev_mer.py           # 支援 --device mps
│   ├── batch_generate.py         # 支援 --repetition_penalty 參數
│   └── finetune_with_eval.py
├── results/                      # 訓練原始歷程、RESULTS_SUMMARY.md、trainer_state.json
├── requirements-mac.txt          # Mac 專用乾淨依賴清單
└── .venv/                        # 本地虛擬環境 (Python 3.12 + MPS PyTorch)
```

---

## 環境配置 (Mac Apple Silicon)

若需在新環境重建：

```bash
# 建立虛擬環境 (Python 3.12)
uv venv .venv --python 3.12
source .venv/bin/activate

# 安裝 Mac 支援的相依套件
uv pip install -r requirements-mac.txt
```

---

## 常用評測指令

所有腳本已更新為**預設自動偵測 Apple Silicon `mps` 硬體加速**。

### 1. 執行 10 句真人 Dev 集評測 (MER)

```bash
.venv/bin/python scripts/eval_dev_mer.py \
  --model_path model/checkpoint-8000-deploy \
  --label dev_8000
```

### 2. 測試 66 句 Benchmark 集 (重現重複退化)

```bash
.venv/bin/python scripts/eval_hotword_testset.py \
  --model_path model/checkpoint-8000-deploy \
  --label repro_8000_vanilla \
  --batch_size 1 \
  --max_audio_duration_sec 1e9
```

### 3. 加入 Repetition Penalty 測試防護效果

實測在 `model.generate` 加入 `repetition_penalty=1.05 ~ 1.1` 可彻底消除 `942.wav` 等音檔的重複死循環：

```bash
.venv/bin/python scripts/eval_hotword_testset.py \
  --model_path model/checkpoint-8000-deploy \
  --label fix_rep_penalty_105 \
  --batch_size 1 \
  --repetition_penalty 1.05 \
  --max_audio_duration_sec 1e9
```

### 4. 執行 Pretrain Baseline 對照組

```bash
.venv/bin/python scripts/eval_hotword_testset.py \
  --model_path model/pretrain-baseline \
  --label baseline_vanilla \
  --batch_size 1 \
  --max_audio_duration_sec 1e9
```
