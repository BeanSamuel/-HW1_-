# 2023 Artificial Intelligence Hw1 -- 多國外幣匯率預測
## 任務
輸入第 1~4 天的8種外幣匯率資料，預測第 5 天8種外幣的"現鈔買入"。
## 資料集
- 訓練資料
從外匯比率網下載的每日外匯資料，共包含8種外幣。
Training data來源時間: 2006/01 ~ 2019/9
- 測試資料
以4天為一組的測試資料，第五天為Label，組別間順序已隨機打亂，並隱藏日期欄位。

## 環境
Python版本: Python 3.8.10
```cmd!
pip install -r requirements.txt
```

## 模型訓練
``` cmd!
python3 main.py
```
- model_name : 在model_chioces中選擇模型
- model_chioces: 調整模型超參數
- save_path : 存模型的路徑和名稱

## 模型預測
```cmd!
python3 predict.py
```
- model_name : 在model_chioces中選擇模型
- model_chioces: 調整模型超參數
- save_path : 結果路徑和名稱

