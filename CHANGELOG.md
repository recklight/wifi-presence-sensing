# 變更紀錄

本檔案記錄 `wifi-presence-sensing` 的所有重要變更。

格式依循 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.1.0/)，
版本號依循 [語意化版本](https://semver.org/lang/zh-TW/)。

## [Unreleased]

---

## [0.1.0] - 2026-10-17

### 新增

- 首次釋出。
- 讀取 Intel 5300 的 beamforming report 擷取檔：向量化解位元封裝、套用接收天線排列、展開 32-bit 時間戳記回繞，CSI 長度不符與截斷的 frame 記錄後跳過；依頻寬、發射鏈數與接收鏈數分組堆疊，`tx_policy` 可選 `dominant` 或 `first_chain`。
- 校正：spatial mapping 還原（`identity`、`hadamard`、`fourier` 或自訂相位表）、以 RSSI／AGC 校正到絕對尺度、三種雜訊模型（`reported`、`sentinel_guard`、`robust`），以及 BPSK、QPSK、16-QAM 與 64-QAM 的 effective SNR。
- 在停頓處切段，每段各自重取樣到均勻網格，並回報這個取樣率能解析哪些動作頻帶。
- EMD、EEMD 與 CEEMDAN，三種可抽換的停止準則；EEMD 的結果只由 seed 決定，與 `n_jobs` 無關。
- 每個視窗四組特徵：接收功率（`power__`）、時間平均振幅（`response__`）、子載波剖面（`profile__`）與各階 IMF 統計量（`modal__`）。
- 人員偵測分類器（logistic regression、random forest、SVM、MLP），評估時與多數類別投票、單一平均振幅特徵、原始振幅特徵三個基線並列。
- 三個指紋定位器（接收功率、原始 CSI 振幅、模態特徵），以 `knn` 或 `probabilistic` 比對 radio map；誤差距離 CDF 附 bootstrap 信賴區間與重複曲線檢查。
- 依場次、錄製檔或時間區塊分組的分層 k 折交叉驗證，可選巢狀搜尋；指標有 UAR、敏感度、特異度、macro F1、混淆矩陣與 ROC-AUC。
- 振幅表 CSV 讀取：驗證欄位，遮罩哨兵值與超出範圍的值並回報比例。
- typer CLI：`inspect`、`synthesize`、`prepare`、`decompose`、`train`、`evaluate`、`predict`、`localize`；`inspect`、`prepare`、`decompose`、`evaluate`、`localize` 每次執行都留下 `config.yaml` 與 `manifest.json`。
- 模擬擷取檔與振幅表，測試、範例與快速開始都用它們，不需要任何量測資料。
- 英文版 README（`README.md`，GitHub 預設顯示），繁體中文版在 `README.zh-TW.md`。
- 流程圖分英文與繁體中文兩版：英文版是 `docs/workflow.html` 與 `docs/images/workflow.png`，繁體中文版是 `docs/workflow.zh-TW.html` 與 `docs/images/workflow.zh-TW.png`。
