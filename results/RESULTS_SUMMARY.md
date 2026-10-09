# checkpoint-8000 重複退化現象：已知結果摘要

這份文件整理跟 `model/checkpoint-8000-deploy/` 這個 checkpoint 直接相關的既有測試結果，
方便你在重現前先知道「預期會看到什麼」，以及我們自己已經做過的分析、排除掉哪些候選解釋。
完整的原始數字在 `trainer_state.json`（本資料夾）裡，這裡只節錄跟診斷「為什麼會重複退化」
直接相關的部分。

## ⚠️ 重要：checkpoint-8000 不是訓練時自動選出的最佳 checkpoint

訓練（`outputs/full_run/1e-5_v2/`，8000 step，每 200 step 存一個 checkpoint）結束時，
`trainer_state.json` 記錄的 `best_model_checkpoint` 其實是 **checkpoint-5800**
（10句真人dev的generative MER 10.25%），不是 checkpoint-8000。

checkpoint-8000 是訓練跑完的最後一步，我們之後選它來打包/測試，單純因為它是「訓練到底」
的版本，是我們自己對外部署/測試最久、資料最完整的版本，不是因為它在任何自動化指標上
表現最好。如果你發現 checkpoint-5800（或其他 step）表現不同，這是預期之內的，
不代表 checkpoint-8000 的數字有誤。`trainer_state.json`裡每個step的三個指標（10句dev
MER／66句keyword recall／66句觀察MER）都有紀錄，**這三個指標之間不一定一致**——同一個
checkpoint在某個指標上表現好，不代表在另一個指標上也最好，我們自己觀察到至少另外兩個
checkpoint（3400、6000）在66句觀察指標上比8000更好，但這份包沒有附上那兩個checkpoint，
只是提醒你不要預設8000是「最好的」版本，它只是「我們最熟悉、測得最完整」的版本。

## 訓練設定

| 項目 | 值 |
|---|---|
| 訓練程式 | `scripts/finetune_with_eval.py`（對應 git commit `ccbf7c9`，LoRA/freeze-encoder 功能尚未加入前的版本） |
| learning_rate | 1e-5 |
| max_steps | 8000 |
| per_device_train_batch_size | 1 |
| gradient_accumulation_steps | 1 |
| max_length | 6144 |
| eval/save 頻率 | 每 200 step |
| save_only_model | True |
| bf16 / gradient_checkpointing | 都開啟 |
| seed | 42 |
| lr_scheduler | linear, warmup_steps=0 |
| weight_decay | 0.0 |

訓練資料：`data/train_full.jsonl`，3226 筆，由 `data/train_audio.tar.gz` 解壓縮後的
13 個語者音檔生成。語者音檔來自同一組 226 筆醫療交班文字（`BlueMagpie-TTS` 合成），
**訓練集裡去掉時間戳/語者標籤後只有 337 種不同文字**，每種文字被 13 個語者各念一次左右
——這是我們後來懷疑「文字/音色多樣性太窄，可能跟重複退化有關」的根據之一，但尚未證實是
唯一或主要原因（見下方「已排除/已確認的候選解釋」）。

官方 repo 版本：`finetune_official/`，對應 commit `61bc29c`。

## ⚠️ 重現時請注意：具體哪幾句退化、重複幾次，每次跑可能不一樣

我們自己在打包完成後，特地重新用一個全新建立的 conda 環境（照這份 README 的步驟從頭
`pip install -r requirements.txt`）跑過一次 `scripts/eval_hotword_testset.py` 驗證。
結果：

- recall（42.53%）、false alarm（8）都跟下面「原始記錄」很接近。
- **但實際退化的音檔ID跟重複次數不一樣**：這次是 942(×568)、939(×264)、884(×184)，
  原始記錄是 503(×1426)、884(×1384)、939(×1865)、940(×1967)。884、939 兩次都退化，
  但 503、940 這次正常，反而多了 942，而且這次重複次數明顯少很多（個位數百次，不是
  上千次），MER（46.68%）也因此比原始記錄（98.57%）低很多。

**結論：這個現象本身（66句裡有少數幾句會嚴重重複）是可以重現的，但具體哪幾句、重複
多嚴重，不是逐字可重現的。** 我們推測是貪婪解碼對浮點數誤差累積很敏感——同一個batch裡
音檔的排列組合（padding長度）不同，就可能讓某個本來接近臨界值的token選擇走向不同分支，
整段生成因此完全不同。這點很重要：**如果你重現時看到的退化ID跟我們下面列的不完全一樣，
不代表打包有問題或你的環境設錯，這是這個現象本身的特性。** 你可以多跑幾次（固定種子下
理論上同一台機器應該要能重現同一個結果，但不同機器/不同批次組合可能不會）、或者改用
`--batch_size 1`逐筆跑（排除掉batch內的padding干擾，看看是否更穩定），這些都是值得你
自己實驗的方向。

## 已知的重複退化案例（三種測試模式，數字不能直接比較，且如上所述不保證逐字重現）

三種模式送進模型的 prompt 不同，不是在比較「哪個數字比較準」，而是呈現「同一個模型在不同
使用情境下穩不穩定」：

- **vanilla**：沒有熱詞提示的裸推論（`eval_dev_mer.py`／`eval_hotword_testset.py` 不加
  `--use_hotword_prompt`）。
- **oracle prompt**：用 `hotwords.json` 的正確答案當熱詞提示（`--use_hotword_prompt`），
  是理想上限情境，不是真實部署的樣子。
- **Production pipeline**：用正式產品的 KWS 真實偵測熱詞 + 後修（`--production_pipeline`），
  比較接近真實部署，但這裡用的是 HuggingFace transformers 模擬（不是真正 vLLM serving，
  兩者生成行為有落差，見下方說明）。

### 10 句 dev（真人錄音）

| 模式 | MER |
|---|---|
| vanilla | 10.25% |
| Production（HF 模擬） | 9.70% |
| **Production（真正 vLLM）** | **9.83%** |
| oracle prompt | 9.40% |

10 句這個小測試集本身沒有出現重複退化案例。

### 66 句 benchmark（熱詞測試集，含真人錄音）

| 模式 | recall | MER（vs 參考文字，非人工真值） | false alarm | 重複退化 |
|---|---|---|---|---|
| vanilla | 42.71% | 98.57% | 7 | **4 句**（原始記錄）：503(×1426)、884(×1384)、939(×1865)、940(×1967)——**重現時實際看到942(×568)、939(×264)、884(×184)，見上方說明** |
| Production（HF 模擬） | 65.55% | 55.55% | 15 | **2 句**：860(×103)、863(×1073) |
| **Production（真正 vLLM）** | 65.38% | **28.57%** | 16 | **0 句**（全部 66 句逐一掃描確認零退化） |
| oracle prompt | 79.61% | 28.66% | 28 | 無 |

**66句的「真正vLLM vs HF模擬」落差，是整個診斷過程最關鍵的發現**：同一個checkpoint、
同樣的860/863兩句音檔，用HuggingFace transformers模擬Production情境會重複退化
（860重複「她就確認，老母」103次／927字，863重複「那我幫他，」1073次／5950字），
但換成**真正的vLLM serving**（`moss-asr-deploy`的vLLM + proxy container），同樣的
音檔完全正常（860變成1110字、863變成1228字，都沒有重複）。這代表至少一部分重複退化
現象，是HuggingFace transformers的`model.generate()`這個推論路徑本身造成的（例如
attention實作、數值精度、貪婪解碼的實作細節跟vLLM不同），不完全是模型權重的問題——
**這也是這次打包的主要目的**：讓你能獨立重現、確認這個落差。

**另外補充一個背景觀察（不在這份包的範圍內，沒有附對應的checkpoint，僅供參考）**：
我們另外測過訓練過程中的checkpoint-6000（66句觀察指標比8000更好的一個step），
它在vanilla模式下比8000穩（1句退化 vs 4句），但在oracle prompt模式下反而比8000更不穩
（2句退化 vs 0句）。這代表「哪個checkpoint比較不會退化」會隨著prompt情境改變，不是
某個checkpoint全面比另一個好，也代表**「給正確熱詞提示能穩定生成」不是通用規律**——
8000的oracle prompt版本這次測起來完全沒有退化，但不能一般化成「給對提示就能解決重複
問題」，比較可能是8000剛好對那組熱詞/那個訓練階段比較穩定，是個別現象不是機制性的修復。

**860 這一句音檔特別容易誘發退化**：在checkpoint-8000(Production/HF模擬)、以及我們
之前測過的其他訓練階段checkpoint都出現過，但每次重複的具體內容不同，說明是這句音檔
本身的聲學特性（可能是雜訊/長靜音/語者交疊）疊加「fine-tune後普遍不穩定」共同造成，
不是單一bug。

### 86 分鐘部署音檔（`server.py` 真實切法，3 段）

- **vanilla**：segment 0（第 62 秒附近）出現嚴重重複退化，拖垮整段約 27 分鐘的輸出。
- **Production（HF 模擬，只測 segment 0）**：同樣在 62 秒附近嚴重退化，重複「我只，」
  約 5034 次，灌滿整段輸出。這個案例**沒有像 66 句的 860/863 那樣因為換真正 vLLM 就消失**
  ——我們還沒有用真正 vLLM 重新測過這支音檔，所以不確定這個特定案例是不是 HF 推論路徑
  造成的，這點留給你驗證。

### after0602（50 個自有真人醫療錄音，`data/after0602.tar.gz`）

全部 50 個檔案都用 Production pipeline（HF 模擬）跑過，原始逐字稿存在
`outputs/deploy_audio_eval/checkpoint-8000_after0602_production/<id>/full_transcript_raw.txt`
這個路徑模式底下（我們自己機器上的路徑，你重新跑的話路徑會不同）。沒有 ground truth，
純粹是人工核對有沒有退化用的。我們還沒有用真正 vLLM 重測這批音檔（只對 66 句的 860/863
做過），所以這批資料主要是給你「大量真實錄音」的素材，不是已經有定論的案例。

## 已排除 / 已確認的候選解釋（避免重複踩同樣的坑）

1. **attention 實作（train 用 sdpa、eval 用未指定）不一致**：已用程式碼直接檢查
   `model.config._attn_implementation` 等欄位，確認載入後三者都是 `sdpa`，**已排除**。
2. **EOS token 沒有被排進訓練 loss**：已讀 `DataCollator` 程式碼確認 EOS 有正確算進
   loss，**已排除**。
3. **訓練 prompt 從未出現「熱詞提示：」這個格式**：已用程式驗證，訓練資料裡 0 次出現
   這個字串，但 chat template 本身在 train/eval/production 三邊共用、一致，**這是真正的
   訓練資料分布缺口，但跟重複退化的因果關係還沒有確認**，只確認會影響熱詞提示情境下的
   recall。
4. **訓練文字/音色多樣性太窄（337 種文字、13 個音色）**：我們量過「輸出跟訓練文字的
   10-字元片段重疊率」，fine-tune 後的模型（約 33%）比 pretrain（約 12%）高出很多，
   顯示 fine-tune 讓模型更依賴背誦訓練內容，但**重複退化的那幾句輸出完全不包含訓練文字**
   （循環的是沒在訓練資料出現過的短片段），所以「背文字」頂多是部分原因，不是重複退化
   本身的直接機制。
5. **多語者/情緒張力/錄音品質**：曾假設重複退化集中在多語者對話，但用 AISHELL-4/AliMeeting
   的多語者樣本測試後發現並不成立（10 句裡只有 1 句輕微重複、且是正常口語停頓）。
   修正後的假設是「情緒張力高/真實場域錄音品質」可能是更相關的因素，但這個方向沒有
   進一步驗證，樣本量也還太小，**這是目前最值得你繼續深入的方向之一**。

## 這個資料夾裡的其他檔案

- `trainer_state.json`：完整訓練過程紀錄，每 200 step 的 train loss / eval_mer /
  66 句觀察指標都在裡面（鍵名 `eval_test_*`）。
- `training_args.bin`：PyTorch pickle 格式的完整 `TrainingArguments`（可用
  `torch.load("training_args.bin", weights_only=False)` 讀取）。
- `all_results.json` / `train_results.json`：訓練結束時的彙總數字（epoch、
  train_loss、train_runtime 等）。
