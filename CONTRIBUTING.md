# 參與開發

歡迎修正錯誤或補上功能，送 pull request 前請先跑過下面的檢查。

---

## 開發環境

需要 Python 3.10 以上。建議使用虛擬環境：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 安裝本專案

```bash
pip install -e ".[dev]"
wifi-presence-sensing --help
```

`dev` extra 包含 pytest、hypothesis 與 ruff。

---

## 提交前的檢查

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

CI 會執行相同的檢查：ruff 與 Python 3.10 語法閘門單獨一個 job，pytest 在 Linux 與 Windows 上跑 Python 3.10–3.13，另外用最舊的相依版本再跑一次。

### Python 3.10 相容性

本專案宣告 `requires-python = ">=3.10"`，請避開 3.11 以後才有的語法與標準函式庫：

| 不要用 | 請用 |
| --- | --- |
| `tomllib` | 本專案的設定檔是 YAML，不需要 TOML |
| `typing.Self` | 類別名稱字串或 `TypeVar` |
| `enum.StrEnum` | `class Foo(str, Enum)` |
| `datetime.UTC` | `datetime.timezone.utc` |
| `except*`、`ExceptionGroup` | 一般的 `except` |
| `asyncio.TaskGroup` | `asyncio.gather` |
| PEP 695 泛型（`class Foo[T]`、`type X = ...`） | `TypeVar` 與 `TypeAlias` |
| `itertools.batched` | 明確的切片迴圈 |

任何 Python 版本都能執行的 3.10 語法閘門：

```bash
python -c "import ast,pathlib; [ast.parse(p.read_text(encoding='utf-8'), str(p), feature_version=(3, 10)) for p in pathlib.Path('.').rglob('*.py') if not {'.venv', 'build', 'dist', '__pycache__'} & set(p.parts)]"
```

`tests/test_python310_compat.py` 會掃描上述語法、3.11 以後才有的標準函式庫名稱與 NumPy 2.0 移除的別名，並檢查 `pyproject.toml` 的相依版本下限。

---

## 程式碼風格

- 每個模組開頭加 `from __future__ import annotations`。
- 公開函式都要有 type hints 與英文 numpy 風格 docstring；實作文獻方法的地方，在 `References` 段標明出處。
- README 有英文（`README.md`）與繁體中文（`README.zh-TW.md`）兩版，改一份就要同步改另一份；本檔與 `CHANGELOG.md` 用繁體中文。
- 用 `logging` 輸出診斷訊息，不要用 `print`。
- 陣列運算盡量向量化，不寫逐元素的 Python 迴圈。
- 參數都放在 `config.py` 的 dataclass，不要在程式裡寫死路徑或數值。

---

## 資料

**本 repository 不包含任何實驗資料。** 測試只使用程式合成的擷取檔、振幅表與特徵表。請不要提交 `.dat` 擷取檔、振幅表 CSV、`.joblib` 模型檔、自己資料集的 manifest，或任何含個人資訊的檔案。`.gitignore` 已經排除這些檔案與 `data/raw/`、`outputs/`，提交前仍請用 `git status` 確認一次。

---

## 提交訊息

請使用 Conventional Commits（`feat`／`fix`／`docs`／`test`／`refactor`）。有功能變動時，請同步更新 `CHANGELOG.md` 的 `[Unreleased]` 段。
