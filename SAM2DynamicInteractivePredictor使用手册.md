# SAM2DynamicInteractivePredictor 使用手册

## 目录
1. [概述](#概述)
2. [核心概念](#核心概念)
3. [初始化與配置](#初始化與配置)
4. [Mask 操作詳解](#mask-操作詳解)
5. [記憶體管理與 update_memory 機制](#記憶體管理與-update_memory-機制)
6. [Frame Cache 機制詳解](#frame-cache-機制詳解)
7. [單個影像中多物件管理](#單個影像中多物件管理)
8. [提示類型混用詳解](#提示類型混用詳解)
9. [Memory Bank 快照與恢復](#memory-bank-快照與恢復)
10. [Memory Bank 完整存儲方案](#memory-bank-完整存儲方案)
11. [數據保存與加載](#數據保存與加載)
12. [完整使用範例](#完整使用範例)
13. [重要注意事項](#重要注意事項)
14. [API 參考](#api-參考)

---

## 概述

`SAM2DynamicInteractivePredictor` 是 SAM2 的進階擴展版本，支持動態交互式物件分割與追蹤。它的核心特性包括：

- **免訓練 (Training-Free)**：直接使用預訓練的 SAM2 模型
- **動態交互**：可在處理過程中隨時新增、更新物件
- **持續學習**：支持對現有物件添加新的提示以改進性能
- **多圖像支持**：可處理獨立圖像序列，實現跨圖像物件追蹤
- **記憶體銀行**：維護物件狀態的記憶體，實現跨幀追蹤

**重要**：這是一個基於記憶體的推理系統，不涉及模型訓練或微調。

---

## 核心概念

### 1. 物件 ID (Object ID) 系統

```python
obj_id: 客戶端物件識別號
   ↓ (通過 obj_id_to_idx 映射)
obj_idx: 模型內部物件索引 (0 到 max_obj_num-1)
```

- **obj_id**：用戶定義的物件標識符 (可以是任意整數 < max_obj_num)
- **obj_idx**：模型內部使用的索引
- **max_obj_num**：最大可追蹤物件數量（初始化時設定，默認為 3）

### 2. 記憶體銀行 (Memory Bank)

記憶體銀行是一個列表，存儲每個處理過的圖像狀態：

```python
memory_bank = [
    {  # 第一幀
        "maskmem_features": Tensor,      # 編碼後的 mask 特徵
        "maskmem_pos_enc": Tensor,       # 位置編碼
        "pred_masks": Tensor,            # 預測的低解析度 masks
        "obj_ptr": Tensor,               # 物件指針嵌入
        "object_score_logits": Tensor    # 物件品質分數
    },
    {  # 第二幀
        ...
    },
    ...
]
```

**關鍵點**：
- 每次調用 `update_memory=True` 時，會向 memory_bank 添加一個新條目
- memory_bank 的長度 = 使用 `update_memory=True` 調用的次數
- 所有幀的記憶體會被串聯起來，用於條件化當前幀的特徵

### 3. 三種提示類型

```python
# 1. 邊界框 (Bounding Boxes)
bboxes = [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]

# 2. 點提示 (Points)
points = [[x, y], [x, y], ...]
labels = [1, 0, 1, ...]  # 1=正向點, 0=負向點

# 3. Mask 提示
masks = [mask1, mask2, ...]  # 每個 mask 形狀為 (H, W)
```

---

## 初始化與配置

### 基本初始化

```python
from ultralytics.models.sam import SAM2DynamicInteractivePredictor

# 配置選項
overrides = dict(
    conf=0.01,              # 信心閾值
    task="segment",         # 任務類型
    mode="predict",         # 模式
    imgsz=1024,            # 圖像大小
    model="sam2_t.pt",     # 模型文件 (sam2_t.pt, sam2_s.pt, sam2_b.pt, sam2_l.pt)
    save=False             # 是否保存結果
)

# 創建預測器
predictor = SAM2DynamicInteractivePredictor(
    overrides=overrides,
    max_obj_num=10         # 最大物件數量
)
```

### 參數說明

| 參數 | 類型 | 默認值 | 說明 |
|------|------|--------|------|
| `cfg` | dict | DEFAULT_CFG | 配置字典 |
| `overrides` | dict | None | 覆蓋默認配置的選項 |
| `max_obj_num` | int | 3 | 最大可追蹤物件數量（固定特徵大小） |
| `_callbacks` | dict | None | 回調函數字典 |

### 可用模型

- `sam2_t.pt` - Tiny 模型（最快，38.9M 參數）
- `sam2_s.pt` - Small 模型
- `sam2_b.pt` - Base 模型（80.8M 參數）
- `sam2_l.pt` - Large 模型（最準確）

---

## Mask 操作詳解

### ⚠️ 重要發現

**SAM2DynamicInteractivePredictor 沒有顯式的 add_mask、delete_mask、modify_mask、clear_mask 方法**。

所有操作都通過 `inference()` 方法和 `obj_ids` 參數來實現。

### 1. 新增物件 (Add Object)

使用 `update_memory=True` 和新的 `obj_id` 來新增物件：

```python
# 方法 1: 使用邊界框新增
predictor(
    source="image1.jpg",
    bboxes=[[100, 100, 200, 200]],  # 邊界框
    obj_ids=[0],                     # 新物件 ID
    update_memory=True               # 更新記憶體
)

# 方法 2: 使用點提示新增
predictor(
    source="image2.jpg",
    points=[[150, 150], [180, 180]],  # 點座標
    labels=[1, 1],                     # 正向點
    obj_ids=[1],                       # 另一個物件 ID
    update_memory=True
)

# 方法 3: 使用 mask 提示新增
import numpy as np
mask = np.zeros((1024, 1024), dtype=bool)
mask[100:200, 100:200] = True

predictor(
    source="image3.jpg",
    masks=[mask],
    obj_ids=[2],
    update_memory=True
)
```

### 2. 修改/精煉物件 (Modify/Refine Object)

使用**相同的 obj_id** 和 `update_memory=True` 來更新物件：

```python
# 第一次：初始定義物件 0
predictor(
    source="image1.jpg",
    bboxes=[[100, 100, 200, 200]],
    obj_ids=[0],
    update_memory=True
)

# 第二次：為物件 0 添加精煉提示（例如物件外觀改變）
predictor(
    source="image5.jpg",
    points=[[150, 150]],      # 新的提示點
    labels=[1],               # 正向點
    obj_ids=[0],              # 相同的 obj_id！
    update_memory=True        # 更新記憶體
)

# 第三次：進一步精煉
predictor(
    source="image8.jpg",
    bboxes=[[120, 120, 220, 220]],  # 更新的邊界框
    obj_ids=[0],                     # 仍然是物件 0
    update_memory=True
)
```

**關鍵點**：
- 每次使用相同的 `obj_id` 調用 `update_memory=True`，都會在 memory_bank 中添加新的記憶體
- 這**不會覆蓋**之前的記憶體，而是**累積**記憶體
- 所有歷史記憶體會共同影響未來的預測

### 3. 只進行推理（不更新記憶體）

```python
# 使用現有記憶體進行推理
results = predictor(source="image10.jpg", update_memory=False)
# 或簡單地：
results = predictor(source="image10.jpg")

# 獲取結果
pred_masks = results[0].masks.data    # 預測的 masks
pred_scores = results[0].boxes.conf   # 信心分數
```

### 4. "刪除"物件 (Delete Object)

**沒有直接的刪除方法**。變通方案：

#### 方法 A：重新創建預測器（完全重置）

```python
# 重新創建預測器（清除所有記憶體）
predictor = SAM2DynamicInteractivePredictor(
    overrides=overrides,
    max_obj_num=10
)
# 重新添加需要的物件
predictor(source="image.jpg", bboxes=[[...]], obj_ids=[0], update_memory=True)
```

#### 方法 B：手動清除記憶體（僅用於高級用戶）

```python
# 清除所有記憶體
predictor.memory_bank = []
predictor.obj_idx_set = set()
predictor.obj_id_to_idx = predictor.obj_idx_to_id = OrderedDict(
    enumerate(range(predictor._max_obj_num))
)

# 或僅清除記憶體銀行但保留物件追蹤
predictor.memory_bank = []
```

#### 方法 C：選擇性過濾結果

```python
# 推理時獲取所有物件
results = predictor(source="image.jpg")

# 僅使用感興趣的物件的 masks
# 物件按照 obj_idx_set 的順序返回
desired_obj_indices = [0, 2]  # 僅要物件 0 和 2
filtered_masks = results[0].masks.data[desired_obj_indices]
```

### 5. 清除所有記憶體 (Clear Memory)

```python
# 完全重置（推薦）
predictor = SAM2DynamicInteractivePredictor(
    overrides=overrides,
    max_obj_num=10
)

# 或手動清除（進階）
predictor.memory_bank = []
predictor.obj_idx_set = set()
from collections import OrderedDict
predictor.obj_id_to_idx = predictor.obj_idx_to_id = OrderedDict(
    enumerate(range(predictor._max_obj_num))
)
```

---

## 記憶體管理與 update_memory 機制

### update_memory=True vs False

```python
# update_memory=True：更新記憶體
# - 必須提供 obj_ids
# - 必須提供至少一種提示（bboxes/points/masks）
# - 會向 memory_bank 添加新條目
# - 返回預測結果
predictor(
    source="image.jpg",
    bboxes=[[100, 100, 200, 200]],
    obj_ids=[0],
    update_memory=True  # 添加到記憶體
)

# update_memory=False（或省略）：僅推理
# - 不需要 obj_ids 或提示
# - 不修改 memory_bank
# - 使用現有記憶體進行預測
# - 返回所有已追蹤物件的結果
results = predictor(
    source="image.jpg",
    update_memory=False  # 或省略此參數
)
```

### 內部工作流程

#### 當 `update_memory=True` 時：

```
1. get_im_features(im)
   └─> 提取圖像特徵（vision_feats, vision_pos_embeds, high_res_features）

2. _prepare_prompts()
   └─> 將提示（bboxes/points/masks）轉換為模型格式

3. update_memory(obj_ids, points, labels, masks)
   ├─> 對每個 obj_id：
   │   ├─> 映射 obj_id → obj_idx
   │   ├─> 添加到 obj_idx_set
   │   └─> 調用 track_step(obj_idx, point, label, mask)
   │       └─> 運行 SAM heads 生成 mask
   │
   ├─> 合併所有物件的輸出
   ├─> 使用 _encode_new_memory() 編碼 masks
   └─> 將 consolidated_out 添加到 memory_bank

4. track_step()
   └─> 使用所有記憶體進行最終預測

5. 返回結果（pred_masks, pred_scores）
```

#### 當 `update_memory=False` 時：

```
1. get_im_features(im)
   └─> 提取圖像特徵

2. track_step()（無 obj_idx 參數）
   ├─> _prepare_memory_conditioned_features(None)
   │   └─> 使用 memory_attention 結合所有記憶體特徵
   │
   └─> _forward_sam_heads() 預測所有物件

3. 過濾並返回已追蹤物件的結果
```

---

## Frame Cache 機制詳解

### 問題：同一張圖像多次輸入，cache 了幾個 frame？

**答案：每次調用 `update_memory=True` 都會添加一個新的 frame 到 memory_bank**。

### 實驗場景

```python
# 場景：同一張圖像 "image.jpg"，逐步添加提示

# 第一次：1 個點
predictor(
    source="image.jpg",
    points=[[100, 100]],
    labels=[1],
    obj_ids=[0],
    update_memory=True
)
print(f"Memory bank 長度: {len(predictor.memory_bank)}")  # 輸出: 1

# 第二次：2 個點（包含第一個點）
predictor(
    source="image.jpg",
    points=[[100, 100], [200, 200]],
    labels=[1, 1],
    obj_ids=[0],  # 相同的 obj_id
    update_memory=True
)
print(f"Memory bank 長度: {len(predictor.memory_bank)}")  # 輸出: 2

# 第三次：3 個點
predictor(
    source="image.jpg",
    points=[[100, 100], [200, 200], [300, 300]],
    labels=[1, 1, 1],
    obj_ids=[0],
    update_memory=True
)
print(f"Memory bank 長度: {len(predictor.memory_bank)}")  # 輸出: 3
```

**結論**：
- ✅ **Memory bank 有 3 個條目**（3 個 frames）
- 每個條目對應一次 `update_memory=True` 調用
- 即使是同一張圖像，每次調用都視為新的 frame
- 所有 3 個 frames 的記憶體會在後續推理中被使用

### 為什麼這樣設計？

這允許**持續學習**：
- 第一次：初步定義物件
- 第二次：添加更多信息（例如物件旋轉了）
- 第三次：進一步精煉（例如遮擋情況）
- 所有這些記憶體共同幫助模型更好地理解物件

### 讀取 Memory Bank 信息

```python
# 查看 memory_bank 長度
num_frames = len(predictor.memory_bank)
print(f"已緩存的 frames 數量: {num_frames}")

# 檢查每個 frame 的內容
for i, frame in enumerate(predictor.memory_bank):
    print(f"\n--- Frame {i} ---")
    print(f"Mask features 形狀: {frame['maskmem_features'].shape}")
    print(f"Predicted masks 形狀: {frame['pred_masks'].shape}")
    print(f"Object scores: {frame['object_score_logits'].flatten()}")

# 查看當前追蹤的物件
print(f"\n已追蹤物件索引: {predictor.obj_idx_set}")
print(f"物件 ID → 索引映射: {predictor.obj_id_to_idx}")

# 獲取特定 frame 的預測 masks
frame_idx = 0
low_res_masks = predictor.memory_bank[frame_idx]['pred_masks']
print(f"Frame {frame_idx} 的低解析度 masks: {low_res_masks.shape}")
# 形狀: (max_obj_num, 1, H/4, W/4)

# 獲取合併的記憶體（所有 frames）
if len(predictor.memory_bank) > 0:
    memory, memory_pos_embed = predictor.get_maskmem_enc()
    print(f"\n合併的記憶體形狀: {memory.shape}")
    print(f"合併的位置編碼形狀: {memory_pos_embed.shape}")
    # memory 形狀: (total_spatial_points, max_obj_num, hidden_dim)
    # total_spatial_points = sum(H_i * W_i for all frames)
```

### Memory Bank 詳細結構

```python
# 每個 frame 在 memory_bank 中的結構：
frame = {
    # 編碼後的 mask 特徵（用於記憶體注意力）
    "maskmem_features": Tensor,
    # 形狀: (max_obj_num, C, H/4, W/4)

    # 位置編碼（用於記憶體注意力）
    "maskmem_pos_enc": List[Tensor],
    # 列表包含多個尺度的位置編碼

    # 預測的低解析度 masks
    "pred_masks": Tensor,
    # 形狀: (max_obj_num, 1, H/4, W/4)
    # 未使用的物件槽填充為 -1024.0

    # 物件指針嵌入（用於記憶體條件化）
    "obj_ptr": Tensor,
    # 形狀: (max_obj_num, hidden_dim)

    # 物件存在性/品質分數
    "object_score_logits": Tensor
    # 形狀: (max_obj_num, 1)
    # 範圍: [-32, 32]，> 0 表示物件存在
}
```

---

## 單個影像中多物件管理

### ✅ 重要發現：可以一次調用更新多個物件！

**答案：是的！可以在單個 frame、單次 predict 調用中同時更新多個物件。**

### 工作原理

從源代碼 `update_memory()` 方法（1858-1896行）可以看到：

```python
for i, obj_id in enumerate(obj_ids):
    # 為每個物件處理提示
    point, label = points[[i]], labels[[i]]
    mask = masks[[i]][None] if masks is not None else None
    out = self.track_step(obj_idx, point, label, mask)
    # 將結果合併到 consolidated_out
    consolidated_out["pred_masks"][obj_idx : obj_idx + 1] = obj_mask
    consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = out["obj_ptr"]

# 最後，整個 consolidated_out 作為一個 frame 被添加
self.memory_bank.append(consolidated_out)
```

**關鍵點**：
- `obj_ids` 可以是多個 ID 的列表
- 每個 `obj_id` 對應一組提示（points、labels、masks）
- 所有物件的輸出被合併到一個 `consolidated_out` 中
- **只添加一個 frame 到 memory_bank**

### 單次調用更新多個物件

#### 範例 1: 使用邊界框定義多個物件

```python
from ultralytics.models.sam import SAM2DynamicInteractivePredictor

# 初始化
overrides = dict(conf=0.01, task="segment", mode="predict", imgsz=1024, model="sam2_b.pt", save=False)
predictor = SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=10)

# ✅ 一次調用同時添加 3 個物件
predictor(
    source="image.jpg",
    bboxes=[
        [100, 100, 200, 200],  # 物件 0 的邊界框
        [300, 300, 450, 450],  # 物件 1 的邊界框
        [500, 100, 650, 250],  # 物件 2 的邊界框
    ],
    obj_ids=[0, 1, 2],  # 對應的物件 IDs
    update_memory=True
)

print(f"Memory bank 大小: {len(predictor.memory_bank)}")  # 輸出: 1
print(f"已追蹤物件: {predictor.obj_idx_set}")  # 輸出: {0, 1, 2}

# 推理會返回所有 3 個物件的 masks
results = predictor(source="image2.jpg")
print(f"檢測到的 masks 數量: {len(results[0].masks.data)}")  # 輸出: 3
```

#### 範例 2: 使用點提示定義多個物件

```python
# ✅ 使用點提示同時定義多個物件
predictor(
    source="image.jpg",
    points=[
        [[150, 150], [160, 160]],  # 物件 0 的點（2個正向點）
        [[350, 350]],               # 物件 1 的點（1個正向點）
        [[550, 150], [600, 200]],  # 物件 2 的點（2個正向點）
    ],
    labels=[
        [1, 1],  # 物件 0 的標籤（都是正向）
        [1],     # 物件 1 的標籤
        [1, 0],  # 物件 2 的標籤（正向+負向）
    ],
    obj_ids=[0, 1, 2],
    update_memory=True
)

print(f"Memory bank 大小: {len(predictor.memory_bank)}")  # 輸出: 1
```

#### 範例 3: 同時更新現有和新增物件

```python
# 第一次：添加物件 0 和 1
predictor(
    source="image1.jpg",
    bboxes=[[100, 100, 200, 200], [300, 300, 400, 400]],
    obj_ids=[0, 1],
    update_memory=True
)
# memory_bank 長度: 1

# 第二次：精煉物件 0，同時新增物件 2
predictor(
    source="image2.jpg",
    bboxes=[
        [110, 110, 210, 210],  # 物件 0 的新位置
        [500, 500, 600, 600],  # 物件 2（新物件）
    ],
    obj_ids=[0, 2],  # 更新物件 0，新增物件 2
    update_memory=True
)
# memory_bank 長度: 2
# 已追蹤物件: {0, 1, 2}

# 推理會返回所有 3 個物件
results = predictor(source="image3.jpg")
print(f"Masks: {len(results[0].masks.data)}")  # 輸出: 3
```

### 數據格式要求

**重要**：提示數據的第一維度必須與 `obj_ids` 的長度匹配。

```python
# ✅ 正確：3 個物件，3 組邊界框
bboxes = [[x1, y1, x2, y2], [x1, y1, x2, y2], [x1, y1, x2, y2]]
obj_ids = [0, 1, 2]

# ✅ 正確：2 個物件，每個有不同數量的點
points = [
    [[100, 100], [110, 110], [120, 120]],  # 物件 0：3 個點
    [[200, 200]],                           # 物件 1：1 個點
]
labels = [
    [1, 1, 0],  # 物件 0 的標籤
    [1],        # 物件 1 的標籤
]
obj_ids = [0, 1]

# ❌ 錯誤：長度不匹配
bboxes = [[x1, y1, x2, y2], [x1, y1, x2, y2]]  # 2 個
obj_ids = [0, 1, 2]  # 3 個
# AssertionError!
```

### 性能優勢

**一次更新多個物件 vs 多次調用**：

```python
# ❌ 方法 A：多次調用（效率較低）
predictor(source="img.jpg", bboxes=[[100, 100, 200, 200]], obj_ids=[0], update_memory=True)
predictor(source="img.jpg", bboxes=[[300, 300, 400, 400]], obj_ids=[1], update_memory=True)
predictor(source="img.jpg", bboxes=[[500, 500, 600, 600]], obj_ids=[2], update_memory=True)
# 結果：memory_bank 有 3 個 frames
# 問題：重複提取圖像特徵 3 次，推理變慢

# ✅ 方法 B：單次調用（推薦）
predictor(
    source="img.jpg",
    bboxes=[
        [100, 100, 200, 200],
        [300, 300, 400, 400],
        [500, 500, 600, 600],
    ],
    obj_ids=[0, 1, 2],
    update_memory=True
)
# 結果：memory_bank 只有 1 個 frame
# 優勢：只提取圖像特徵 1 次，推理更快
```

**性能比較**：

| 方法 | Memory Bank 大小 | 圖像特徵提取次數 | 推理速度 |
|------|-----------------|----------------|---------|
| 多次調用（方法 A） | N frames | N 次 | 慢（線性增長） |
| 單次調用（方法 B） | 1 frame | 1 次 | 快 |

### 最佳實踐

✅ **推薦做法**：
```python
# 1. 同一張圖像中的多個物件，一次性定義
predictor(
    source="image.jpg",
    bboxes=[...所有物件的 bboxes...],
    obj_ids=[0, 1, 2, ...],
    update_memory=True
)

# 2. 新圖像中的新物件，也盡量批量添加
predictor(
    source="image5.jpg",
    bboxes=[...新物件的 bboxes...],
    obj_ids=[3, 4],  # 新物件
    update_memory=True
)
```

❌ **避免**：
```python
# 不要對同一張圖像多次調用 update_memory=True（除非真的需要累積記憶體）
for obj_id in [0, 1, 2]:
    predictor(source="same_image.jpg", bboxes=[...], obj_ids=[obj_id], update_memory=True)
```

### 實用範例：視頻第一幀標註多個物件

```python
import cv2
from ultralytics.models.sam import SAM2DynamicInteractivePredictor

# 初始化
overrides = dict(conf=0.01, task="segment", mode="predict", imgsz=1024, model="sam2_b.pt", save=False)
predictor = SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=10)

# 讀取視頻第一幀
cap = cv2.VideoCapture("video.mp4")
ret, first_frame = cap.read()

# 在第一幀同時標註所有要追蹤的物件
predictor(
    source=first_frame,
    bboxes=[
        [100, 100, 300, 300],  # 人物 1
        [400, 200, 600, 500],  # 人物 2
        [700, 300, 900, 600],  # 車輛
        [50, 50, 150, 150],    # 其他物件
    ],
    obj_ids=[0, 1, 2, 3],  # 4 個物件
    update_memory=True
)

print(f"初始化完成，memory_bank 大小: {len(predictor.memory_bank)}")  # 1
print(f"開始追蹤 {len(predictor.obj_idx_set)} 個物件")  # 4

# 處理後續幀
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 推理所有物件（無需重複標註）
    results = predictor(source=frame)

    # 獲取所有 masks 並可視化
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        for i, mask in enumerate(masks):
            # 繪製每個物件的 mask
            pass

cap.release()
```

---

## 提示類型混用詳解

### 核心問題：不同提示類型能否混用？

**答案：可以！但有特定規則。**

### 提示類型轉換機制

從源代碼（`_prepare_prompts` 方法，762-773行）可以看到：

```python
# Bounding boxes 會被轉換為特殊的點提示
if bboxes is not None:
    bboxes = bboxes.view(-1, 2, 2)  # 轉換為 2 個點
    bbox_labels = torch.tensor([[2, 3]], ...)  # 特殊標籤 2, 3

    # 合併 bboxes 和 points
    if points is not None:
        points = torch.cat([bboxes, points], dim=1)  # 連接
        labels = torch.cat([bbox_labels, labels], dim=1)
    else:
        points, labels = bboxes, bbox_labels
```

**關鍵發現**：
- **Bboxes 本質上是 Points**：每個 bbox 被轉換為 2 個特殊點（標籤 2 和 3）
- **Bboxes 和 Points 可以混用**：它們會被連接在一起
- **Masks 是獨立的**：Masks 不會與 points 合併，而是作為單獨的輸入

### 混用規則

#### ✅ 支持的混用組合

| 組合 | 支持 | 說明 |
|------|------|------|
| **Bboxes + Points** | ✅ | 同一物件可以同時有 bbox 和 points |
| **Bboxes + Masks** | ✅ | 同一物件可以同時有 bbox 和 mask |
| **Points + Masks** | ✅ | 同一物件可以同時有 points 和 mask |
| **Bboxes + Points + Masks** | ✅ | 同一物件可以三者都有 |
| **多物件不同提示** | ⚠️ | 需要特殊處理（見下文） |

### 單個物件混用多種提示

#### 範例 1: Bbox + Points（精煉 bbox）

```python
# 使用 bbox 定義大致區域，用 points 精煉
predictor(
    source="image.jpg",
    bboxes=[[100, 100, 300, 300]],     # 大致區域
    points=[[[200, 200], [250, 250]]], # 精煉點（注意維度！）
    labels=[[1, 0]],                   # 正向+負向點
    obj_ids=[0],
    update_memory=True
)
```

**注意維度**：
- `bboxes`: 形狀 `(N, 4)` → N 個物件
- `points`: 形狀 `(N, M, 2)` → N 個物件，每個有 M 個點
- `labels`: 形狀 `(N, M)` → N 個物件，每個有 M 個標籤

```python
# ✅ 正確：1 個物件，bbox + 2 個 points
bboxes = [[100, 100, 300, 300]]        # (1, 4)
points = [[[200, 200], [250, 250]]]    # (1, 2, 2) - 注意三層嵌套！
labels = [[1, 0]]                       # (1, 2)
obj_ids = [0]

# ❌ 錯誤：維度不匹配
bboxes = [[100, 100, 300, 300]]        # (1, 4)
points = [[200, 200], [250, 250]]      # (2, 2) - 會被當作 2 個物件！
labels = [1, 0]
obj_ids = [0]  # 長度不匹配！
```

#### 範例 2: Points + Mask

```python
import numpy as np

# 創建一個 mask
mask = np.zeros((1024, 1024), dtype=np.float32)
mask[100:300, 100:300] = 1.0

# 使用 mask 定義大致區域，用 points 精煉邊界
predictor(
    source="image.jpg",
    masks=[mask],
    points=[[[150, 150], [250, 250]]], # 精煉點
    labels=[[1, 1]],
    obj_ids=[0],
    update_memory=True
)
```

### 多物件不同提示類型

這是最複雜的場景。**問題**：不同物件使用不同提示類型時如何處理？

#### ⚠️ 限制：無法直接混用

```python
# ❌ 這樣不行：物件 0 用 bbox，物件 1 用 points
predictor(
    source="image.jpg",
    bboxes=[[100, 100, 200, 200]],  # 只有 1 個 bbox（物件 0？）
    points=[[[300, 300]]],          # 只有 1 組 points（物件 1？）
    obj_ids=[0, 1],                 # 2 個物件
    update_memory=True
)
# 這會導致維度不匹配或意外行為！
```

#### ✅ 解決方案 1: 分別調用

```python
# 物件 0 使用 bbox
predictor(
    source="image.jpg",
    bboxes=[[100, 100, 200, 200]],
    obj_ids=[0],
    update_memory=True
)

# 物件 1 使用 points（同一張圖像，第二次調用）
predictor(
    source="image.jpg",
    points=[[[300, 300], [350, 350]]],
    labels=[[1, 1]],
    obj_ids=[1],
    update_memory=True
)

# 結果：memory_bank 有 2 個 frames（但都是同一張圖）
```

**權衡**：
- ✅ 靈活性高，可以為每個物件使用不同提示
- ❌ memory_bank 會有多個 frames（影響性能）
- ❌ 重複提取圖像特徵

#### ✅ 解決方案 2: 統一為 Points（推薦）

```python
# 將所有提示統一為 points 格式

# 物件 0：從 bbox 手動轉換為 points
bbox_0 = [100, 100, 200, 200]
bbox_points_0 = [
    [bbox_0[0], bbox_0[1]],  # 左上角
    [bbox_0[2], bbox_0[3]],  # 右下角
]
bbox_labels_0 = [2, 3]  # SAM2 的 bbox 標籤

# 物件 1：原本就是 points
obj_1_points = [[300, 300], [350, 350]]
obj_1_labels = [1, 1]

# 合併
all_points = [bbox_points_0, obj_1_points]
all_labels = [bbox_labels_0, obj_1_labels]

# 一次調用更新兩個物件
predictor(
    source="image.jpg",
    points=all_points,
    labels=all_labels,
    obj_ids=[0, 1],
    update_memory=True
)

# 結果：memory_bank 只有 1 個 frame！
```

#### ✅ 解決方案 3: 使用佔位符

對於不需要某種提示的物件，使用空的佔位符：

```python
# 物件 0 使用 bbox，物件 1 使用 points，物件 2 使用 mask

import numpy as np

# 準備提示
bboxes = [
    [100, 100, 200, 200],  # 物件 0 的 bbox
    [300, 300, 400, 400],  # 物件 1 的 bbox（作為佔位，也可以用）
    [500, 500, 600, 600],  # 物件 2 的 bbox（作為佔位）
]

# 或者，統一轉換為 points
points = [
    [[100, 100], [200, 200]],  # 物件 0（從 bbox 轉換）
    [[350, 350]],              # 物件 1（真正的 point）
    [[550, 550]],              # 物件 2（佔位或輔助）
]
labels = [
    [2, 3],  # 物件 0（bbox 標籤）
    [1],     # 物件 1（正向點）
    [1],     # 物件 2（正向點）
]

# Mask（可選）
mask_2 = np.zeros((1024, 1024), dtype=np.float32)
mask_2[500:600, 500:600] = 1.0
masks = [
    None,    # 物件 0 沒有 mask
    None,    # 物件 1 沒有 mask
    mask_2,  # 物件 2 有 mask
]

# ❌ 問題：masks 不能包含 None
# ✅ 解決：只為有 mask 的物件單獨調用

# 先用 points 定義物件 0 和 1
predictor(
    source="image.jpg",
    points=[points[0], points[1]],
    labels=[labels[0], labels[1]],
    obj_ids=[0, 1],
    update_memory=True
)

# 再用 mask + point 定義物件 2
predictor(
    source="image.jpg",
    masks=[mask_2],
    points=[points[2]],
    labels=[labels[2]],
    obj_ids=[2],
    update_memory=True
)
```

### 更新時的提示混用

**問題**：更新物件時能否改變提示類型？

**答案**：可以！

```python
# 第一次：用 bbox 定義物件 0
predictor(
    source="image1.jpg",
    bboxes=[[100, 100, 200, 200]],
    obj_ids=[0],
    update_memory=True
)

# 第二次：用 points 精煉物件 0
predictor(
    source="image2.jpg",
    points=[[[150, 150], [180, 180]]],
    labels=[[1, 0]],
    obj_ids=[0],  # 同一個物件
    update_memory=True
)

# 第三次：用 mask 進一步精煉物件 0
import numpy as np
mask = np.zeros((1024, 1024), dtype=np.float32)
mask[120:180, 120:180] = 1.0

predictor(
    source="image3.jpg",
    masks=[mask],
    obj_ids=[0],
    update_memory=True
)

# 所有這些都會累積到 memory_bank 中
print(f"Memory bank 大小: {len(predictor.memory_bank)}")  # 3
```

### 提示類型選擇指南

| 提示類型 | 適用場景 | 優點 | 缺點 |
|---------|---------|------|------|
| **Bounding Box** | 物件位置已知，形狀規則 | 快速標註，適合矩形物件 | 不適合不規則形狀 |
| **Points** | 需要精確控制，處理遮擋 | 靈活，可正向/負向點 | 需要更多手動標註 |
| **Masks** | 已有預分割結果，不規則形狀 | 最精確 | 需要額外的 mask 生成步驟 |
| **混用** | 需要精煉或處理複雜場景 | 結合優點，提高準確性 | 複雜度增加 |

### 實用範例：混合標註工作流

```python
from ultralytics.models.sam import SAM2DynamicInteractivePredictor
import numpy as np

# 初始化
overrides = dict(conf=0.01, task="segment", mode="predict", imgsz=1024, model="sam2_b.pt", save=False)
predictor = SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=10)

# 場景：標註一張包含多個物件的圖像
# - 物件 0（貓）：形狀不規則，使用 points
# - 物件 1（書）：矩形，使用 bbox
# - 物件 2（杯子）：已有預分割 mask

# 方法：統一為 points，但保留 mask
cat_points = [[150, 200], [180, 250], [160, 220]]
cat_labels = [1, 1, 0]  # 2 個正向點，1 個負向點

book_bbox = [300, 100, 450, 250]
book_points = [[book_bbox[0], book_bbox[1]], [book_bbox[2], book_bbox[3]]]
book_labels = [2, 3]  # bbox 標籤

cup_mask = np.load("cup_mask.npy")  # 已有的 mask
cup_points = [[550, 300]]  # 補充一個點來精煉
cup_labels = [1]

# 一次性標註（除了 mask）
predictor(
    source="image.jpg",
    points=[cat_points, book_points, cup_points],
    labels=[cat_labels, book_labels, cup_labels],
    obj_ids=[0, 1, 2],
    update_memory=True
)

# 為物件 2 添加 mask 進行精煉（可選）
predictor(
    source="image.jpg",
    masks=[cup_mask],
    obj_ids=[2],
    update_memory=True
)

print(f"完成標註，memory_bank 大小: {len(predictor.memory_bank)}")  # 2
print(f"已追蹤物件: {predictor.obj_idx_set}")  # {0, 1, 2}

# 在新圖像中追蹤
results = predictor(source="image2.jpg")
```

### 總結：提示混用最佳實踐

✅ **推薦**：
1. 同一物件混用提示（bbox + points 精煉）
2. 統一轉換為 points 格式進行批量處理
3. 根據物件特性選擇合適的提示類型

⚠️ **注意**：
1. 嚴格匹配提示數據的維度
2. 多物件不同提示需要特殊處理
3. 考慮性能影響（減少 memory_bank 大小）

❌ **避免**：
1. 直接混用不同維度的提示
2. 對同一圖像過多次調用 update_memory=True

---

## Memory Bank 快照與恢復

### 使用場景

在訓練或調整過程中，您可能需要：

1. **保存檢查點**：在第 6 個 frame 之前保存 Memory Bank 狀態
2. **實驗多種調整**：第 6 張圖多次調整，每次從同一快照開始
3. **回滾機制**：如果調整效果不好，恢復到之前的狀態

### 快照系統實現

#### 完整快照類

```python
import copy
import pickle
from collections import OrderedDict
import torch

class MemoryBankSnapshot:
    """Memory Bank 快照管理器"""

    def __init__(self, predictor):
        """
        初始化快照管理器

        Args:
            predictor: SAM2DynamicInteractivePredictor 實例
        """
        self.predictor = predictor
        self.snapshots = {}  # 存儲多個命名快照

    def save_snapshot(self, name="default"):
        """
        保存當前 Memory Bank 狀態為快照

        Args:
            name (str): 快照名稱

        Returns:
            dict: 快照信息
        """
        snapshot = {
            # 深拷貝 memory_bank（包含 tensors）
            'memory_bank': self._deep_copy_memory_bank(self.predictor.memory_bank),
            # 拷貝物件追蹤信息
            'obj_idx_set': copy.deepcopy(self.predictor.obj_idx_set),
            'obj_id_to_idx': copy.deepcopy(self.predictor.obj_id_to_idx),
            'obj_idx_to_id': copy.deepcopy(self.predictor.obj_idx_to_id),
            # 元數據
            'max_obj_num': self.predictor._max_obj_num,
            'num_frames': len(self.predictor.memory_bank),
        }

        self.snapshots[name] = snapshot

        print(f"✅ 快照 '{name}' 已保存")
        print(f"   - Frames: {snapshot['num_frames']}")
        print(f"   - 物件: {snapshot['obj_idx_set']}")

        return snapshot

    def restore_snapshot(self, name="default"):
        """
        恢復到指定快照狀態

        Args:
            name (str): 快照名稱

        Raises:
            KeyError: 如果快照不存在
        """
        if name not in self.snapshots:
            raise KeyError(f"快照 '{name}' 不存在。可用快照: {list(self.snapshots.keys())}")

        snapshot = self.snapshots[name]

        # 驗證配置
        if self.predictor._max_obj_num != snapshot['max_obj_num']:
            raise ValueError(
                f"max_obj_num 不匹配: 當前 {self.predictor._max_obj_num}, "
                f"快照 {snapshot['max_obj_num']}"
            )

        # 恢復狀態
        self.predictor.memory_bank = self._deep_copy_memory_bank(snapshot['memory_bank'])
        self.predictor.obj_idx_set = copy.deepcopy(snapshot['obj_idx_set'])
        self.predictor.obj_id_to_idx = copy.deepcopy(snapshot['obj_id_to_idx'])
        self.predictor.obj_idx_to_id = copy.deepcopy(snapshot['obj_idx_to_id'])

        print(f"✅ 已恢復快照 '{name}'")
        print(f"   - Frames: {len(self.predictor.memory_bank)}")
        print(f"   - 物件: {self.predictor.obj_idx_set}")

    def _deep_copy_memory_bank(self, memory_bank):
        """深拷貝 memory_bank（包括 tensors）"""
        copied_bank = []
        for frame in memory_bank:
            copied_frame = {}
            for key, value in frame.items():
                if isinstance(value, torch.Tensor):
                    # Tensor 需要 clone
                    copied_frame[key] = value.clone()
                elif isinstance(value, list):
                    # 列表中的 Tensors
                    copied_frame[key] = [
                        v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                        for v in value
                    ]
                else:
                    copied_frame[key] = copy.deepcopy(value)
            copied_bank.append(copied_frame)
        return copied_bank

    def list_snapshots(self):
        """列出所有快照"""
        print(f"\n📸 快照列表（共 {len(self.snapshots)} 個）:")
        for name, snapshot in self.snapshots.items():
            print(f"  - '{name}': {snapshot['num_frames']} frames, "
                  f"物件 {snapshot['obj_idx_set']}")

    def delete_snapshot(self, name):
        """刪除指定快照"""
        if name in self.snapshots:
            del self.snapshots[name]
            print(f"✅ 快照 '{name}' 已刪除")
        else:
            print(f"❌ 快照 '{name}' 不存在")

    def save_to_file(self, filepath, snapshot_name="default"):
        """將快照保存到文件"""
        if snapshot_name not in self.snapshots:
            raise KeyError(f"快照 '{snapshot_name}' 不存在")

        with open(filepath, 'wb') as f:
            pickle.dump(self.snapshots[snapshot_name], f)

        print(f"✅ 快照 '{snapshot_name}' 已保存到 {filepath}")

    def load_from_file(self, filepath, snapshot_name="loaded"):
        """從文件加載快照"""
        with open(filepath, 'rb') as f:
            snapshot = pickle.load(f)

        self.snapshots[snapshot_name] = snapshot
        print(f"✅ 快照已從 {filepath} 加載為 '{snapshot_name}'")
        print(f"   - Frames: {snapshot['num_frames']}")
        print(f"   - 物件: {snapshot['obj_idx_set']}")

        return snapshot
```

### 使用範例 1: 基本快照和恢復

```python
from ultralytics.models.sam import SAM2DynamicInteractivePredictor

# 初始化預測器
overrides = dict(conf=0.01, task="segment", mode="predict", imgsz=1024, model="sam2_b.pt", save=False)
predictor = SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=10)

# 創建快照管理器
snapshot_manager = MemoryBankSnapshot(predictor)

# 處理前 5 個 frames
for i in range(5):
    predictor(
        source=f"frame_{i:04d}.jpg",
        bboxes=[[100, 100, 200, 200]],
        obj_ids=[0],
        update_memory=True
    )

print(f"已處理 5 個 frames，memory_bank 大小: {len(predictor.memory_bank)}")

# 保存快照（在第 6 個 frame 之前）
snapshot_manager.save_snapshot(name="before_frame_6")

# 現在嘗試第 6 個 frame 的不同調整

# 嘗試 1：使用 bbox
predictor(
    source="frame_0006.jpg",
    bboxes=[[110, 110, 210, 210]],
    obj_ids=[0],
    update_memory=True
)
results_1 = predictor(source="frame_0007.jpg")
print(f"嘗試 1 完成，memory_bank 大小: {len(predictor.memory_bank)}")  # 6

# 恢復到快照
snapshot_manager.restore_snapshot("before_frame_6")
print(f"已恢復，memory_bank 大小: {len(predictor.memory_bank)}")  # 5

# 嘗試 2：使用 points
predictor(
    source="frame_0006.jpg",
    points=[[[150, 150], [180, 180]]],
    labels=[[1, 1]],
    obj_ids=[0],
    update_memory=True
)
results_2 = predictor(source="frame_0007.jpg")
print(f"嘗試 2 完成，memory_bank 大小: {len(predictor.memory_bank)}")  # 6

# 恢復到快照
snapshot_manager.restore_snapshot("before_frame_6")

# 嘗試 3：使用 mask
import numpy as np
mask = np.zeros((1024, 1024), dtype=np.float32)
mask[110:210, 110:210] = 1.0

predictor(
    source="frame_0006.jpg",
    masks=[mask],
    obj_ids=[0],
    update_memory=True
)
results_3 = predictor(source="frame_0007.jpg")
print(f"嘗試 3 完成，memory_bank 大小: {len(predictor.memory_bank)}")  # 6

# 比較結果，選擇最好的...
```

### 使用範例 2: 多個快照管理

```python
# 初始化
predictor = SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=10)
snapshot_manager = MemoryBankSnapshot(predictor)

# Frame 0-2: 初始標註
for i in range(3):
    predictor(source=f"frame_{i}.jpg", bboxes=[[100, 100, 200, 200]], obj_ids=[0], update_memory=True)

snapshot_manager.save_snapshot(name="checkpoint_frame_2")

# Frame 3-5: 添加第二個物件
for i in range(3, 6):
    predictor(source=f"frame_{i}.jpg", bboxes=[[300, 300, 400, 400]], obj_ids=[1], update_memory=True)

snapshot_manager.save_snapshot(name="checkpoint_frame_5")

# Frame 6-8: 添加第三個物件
for i in range(6, 9):
    predictor(source=f"frame_{i}.jpg", bboxes=[[500, 500, 600, 600]], obj_ids=[2], update_memory=True)

snapshot_manager.save_snapshot(name="checkpoint_frame_8")

# 查看所有快照
snapshot_manager.list_snapshots()
# 輸出:
# 📸 快照列表（共 3 個）:
#   - 'checkpoint_frame_2': 3 frames, 物件 {0}
#   - 'checkpoint_frame_5': 6 frames, 物件 {0, 1}
#   - 'checkpoint_frame_8': 9 frames, 物件 {0, 1, 2}

# 恢復到 frame 5 的狀態
snapshot_manager.restore_snapshot("checkpoint_frame_5")
print(f"當前 frames: {len(predictor.memory_bank)}")  # 6
print(f"當前物件: {predictor.obj_idx_set}")  # {0, 1}

# 從這裡開始不同的分支調整...
```

### 使用範例 3: 快照持久化

```python
# 保存快照到文件
snapshot_manager.save_snapshot(name="my_snapshot")
snapshot_manager.save_to_file("snapshots/snapshot_frame_5.pkl", "my_snapshot")

# 稍後，在新的會話中加載
new_predictor = SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=10)
new_snapshot_manager = MemoryBankSnapshot(new_predictor)

# 從文件加載快照
new_snapshot_manager.load_from_file("snapshots/snapshot_frame_5.pkl", "loaded_snapshot")

# 恢復狀態
new_snapshot_manager.restore_snapshot("loaded_snapshot")

# 繼續處理
results = new_predictor(source="new_frame.jpg")
```

### 快照比較工具

```python
def compare_snapshots(snapshot_manager, name1, name2):
    """比較兩個快照的差異"""
    snap1 = snapshot_manager.snapshots[name1]
    snap2 = snapshot_manager.snapshots[name2]

    print(f"\n📊 比較快照 '{name1}' 和 '{name2}':")
    print(f"  Frames: {snap1['num_frames']} vs {snap2['num_frames']}")
    print(f"  物件: {snap1['obj_idx_set']} vs {snap2['obj_idx_set']}")

    # 新增的物件
    new_objects = snap2['obj_idx_set'] - snap1['obj_idx_set']
    if new_objects:
        print(f"  ➕ 新增物件: {new_objects}")

    # 移除的物件
    removed_objects = snap1['obj_idx_set'] - snap2['obj_idx_set']
    if removed_objects:
        print(f"  ➖ 移除物件: {removed_objects}")

# 使用
compare_snapshots(snapshot_manager, "checkpoint_frame_2", "checkpoint_frame_5")
# 輸出:
# 📊 比較快照 'checkpoint_frame_2' 和 'checkpoint_frame_5':
#   Frames: 3 vs 6
#   物件: {0} vs {0, 1}
#   ➕ 新增物件: {1}
```

### 快照策略建議

**何時創建快照**：

✅ **推薦時機**：
1. 每 N 個 frames（例如每 10 frames）
2. 添加新物件之前
3. 進行大量調整之前
4. 達到滿意效果時（作為檢查點）

```python
# 自動快照策略
def process_with_auto_snapshots(predictor, frame_paths, snapshot_interval=10):
    """帶自動快照的處理流程"""
    snapshot_manager = MemoryBankSnapshot(predictor)

    for i, frame_path in enumerate(frame_paths):
        # 處理 frame
        results = predictor(source=frame_path)

        # 每隔 N 幀保存快照
        if (i + 1) % snapshot_interval == 0:
            snapshot_name = f"auto_snapshot_frame_{i+1}"
            snapshot_manager.save_snapshot(snapshot_name)

    return snapshot_manager

# 使用
frame_paths = [f"frame_{i:04d}.jpg" for i in range(100)]
snapshot_manager = process_with_auto_snapshots(predictor, frame_paths, snapshot_interval=10)
snapshot_manager.list_snapshots()
# 會有: auto_snapshot_frame_10, auto_snapshot_frame_20, ..., auto_snapshot_frame_100
```

---

## Memory Bank 完整存儲方案

### 為什麼需要完整存儲？

1. **長期保存**：保存整個推理會話的狀態
2. **跨會話恢復**：在不同時間或機器上繼續工作
3. **實驗追蹤**：保存不同配置的結果以供比較
4. **生產部署**：保存訓練好的記憶體供在線推理使用

### 完整存儲方案

#### 存儲內容清單

需要保存的完整狀態：

```python
{
    # 1. Memory Bank（核心）
    'memory_bank': [...],  # 所有 frame 的記憶體

    # 2. 物件追蹤狀態
    'obj_idx_set': {...},
    'obj_id_to_idx': {...},
    'obj_idx_to_id': {...},

    # 3. 配置信息
    'max_obj_num': int,
    'imgsz': tuple,
    'model_name': str,
    'device': str,

    # 4. 元數據（可選但推薦）
    'metadata': {
        'created_at': str,
        'num_frames': int,
        'num_objects': int,
        'ultralytics_version': str,
        'description': str,
    }
}
```

#### 完整存儲類實現

```python
import pickle
import torch
import json
from pathlib import Path
from datetime import datetime
import ultralytics

class MemoryBankStorage:
    """Memory Bank 完整存儲和加載系統"""

    @staticmethod
    def save(predictor, filepath, description="", metadata=None):
        """
        保存完整的 Memory Bank 狀態

        Args:
            predictor: SAM2DynamicInteractivePredictor 實例
            filepath (str): 保存路徑
            description (str): 狀態描述
            metadata (dict): 額外的元數據
        """
        # 準備保存數據
        save_data = {
            # 核心數據
            'memory_bank': predictor.memory_bank,
            'obj_idx_set': predictor.obj_idx_set,
            'obj_id_to_idx': predictor.obj_id_to_idx,
            'obj_idx_to_id': predictor.obj_idx_to_id,

            # 配置
            'max_obj_num': predictor._max_obj_num,
            'imgsz': predictor.imgsz,
            'model_name': predictor.model.model_name if hasattr(predictor.model, 'model_name') else "unknown",
            'device': str(predictor.device),
            'non_overlap_masks': predictor.non_overlap_masks,

            # 元數據
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'num_frames': len(predictor.memory_bank),
                'num_objects': len(predictor.obj_idx_set),
                'ultralytics_version': ultralytics.__version__,
                'description': description,
                **(metadata or {}),
            }
        }

        # 保存
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 同時保存 JSON 元數據（便於查看）
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            # 只保存元數據和配置（不包括 tensors）
            json_data = {
                'metadata': save_data['metadata'],
                'config': {
                    'max_obj_num': save_data['max_obj_num'],
                    'imgsz': save_data['imgsz'],
                    'model_name': save_data['model_name'],
                    'device': save_data['device'],
                },
                'objects': list(save_data['obj_idx_set']),
            }
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Memory Bank 已保存到 {filepath}")
        print(f"   - Frames: {save_data['metadata']['num_frames']}")
        print(f"   - 物件: {save_data['metadata']['num_objects']}")
        print(f"   - 大小: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"   - 元數據: {metadata_path}")

        return save_data

    @staticmethod
    def load(predictor, filepath, strict=True):
        """
        加載完整的 Memory Bank 狀態

        Args:
            predictor: SAM2DynamicInteractivePredictor 實例
            filepath (str): 文件路徑
            strict (bool): 是否嚴格檢查配置匹配

        Returns:
            dict: 加載的元數據
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        # 加載數據
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        # 驗證配置
        if strict:
            if predictor._max_obj_num != save_data['max_obj_num']:
                raise ValueError(
                    f"max_obj_num 不匹配: 當前 {predictor._max_obj_num}, "
                    f"保存的 {save_data['max_obj_num']}"
                )

            if tuple(predictor.imgsz) != tuple(save_data['imgsz']):
                print(f"⚠️ 警告: imgsz 不匹配: 當前 {predictor.imgsz}, 保存的 {save_data['imgsz']}")

        # 恢復狀態
        predictor.memory_bank = save_data['memory_bank']
        predictor.obj_idx_set = save_data['obj_idx_set']
        predictor.obj_id_to_idx = save_data['obj_id_to_idx']
        predictor.obj_idx_to_id = save_data['obj_idx_to_id']

        if 'non_overlap_masks' in save_data:
            predictor.non_overlap_masks = save_data['non_overlap_masks']

        # 處理設備遷移
        target_device = predictor.device
        saved_device = save_data['device']

        if str(target_device) != saved_device:
            print(f"⚠️ 設備遷移: {saved_device} → {target_device}")
            MemoryBankStorage._move_memory_bank_to_device(predictor.memory_bank, target_device)

        metadata = save_data['metadata']
        print(f"✅ Memory Bank 已從 {filepath} 加載")
        print(f"   - 創建時間: {metadata['created_at']}")
        print(f"   - Frames: {metadata['num_frames']}")
        print(f"   - 物件: {metadata['num_objects']}")
        print(f"   - 描述: {metadata.get('description', 'N/A')}")

        return metadata

    @staticmethod
    def _move_memory_bank_to_device(memory_bank, device):
        """將 memory_bank 中的所有 tensors 移動到指定設備"""
        for frame in memory_bank:
            for key, value in frame.items():
                if isinstance(value, torch.Tensor):
                    frame[key] = value.to(device)
                elif isinstance(value, list):
                    frame[key] = [
                        v.to(device) if isinstance(v, torch.Tensor) else v
                        for v in value
                    ]

    @staticmethod
    def get_info(filepath):
        """獲取保存文件的元數據（無需加載完整數據）"""
        metadata_path = Path(filepath).with_suffix('.json')

        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Fallback: 加載 pickle 文件
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            return save_data.get('metadata', {})
```

### 使用範例 1: 基本保存和加載

```python
from ultralytics.models.sam import SAM2DynamicInteractivePredictor

# 初始化並處理一些 frames
overrides = dict(conf=0.01, task="segment", mode="predict", imgsz=1024, model="sam2_b.pt", save=False)
predictor = SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=10)

# 處理數據
for i in range(10):
    predictor(source=f"frame_{i}.jpg", bboxes=[[100, 100, 200, 200]], obj_ids=[0], update_memory=True)

# 保存
MemoryBankStorage.save(
    predictor,
    filepath="saved_states/my_project.pkl",
    description="10 frames processed, tracking 1 object"
)

# --- 稍後，在新會話中 ---

# 創建新的預測器（相同配置）
new_predictor = SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=10)

# 加載保存的狀態
metadata = MemoryBankStorage.load(new_predictor, "saved_states/my_project.pkl")

# 繼續處理
results = new_predictor(source="frame_11.jpg")
print(f"檢測到 {len(results[0].masks.data)} 個物件")
```

### 使用範例 2: 多版本管理

```python
# 保存不同版本的狀態

# 版本 1: 初始標註
predictor_v1 = SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=10)
for i in range(5):
    predictor_v1(source=f"frame_{i}.jpg", bboxes=[[100, 100, 200, 200]], obj_ids=[0], update_memory=True)

MemoryBankStorage.save(
    predictor_v1,
    "versions/v1_initial.pkl",
    description="Initial annotation with 1 object",
    metadata={'version': 'v1', 'author': 'user1'}
)

# 版本 2: 添加第二個物件
predictor_v2 = SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=10)
MemoryBankStorage.load(predictor_v2, "versions/v1_initial.pkl")

for i in range(5, 10):
    predictor_v2(source=f"frame_{i}.jpg", bboxes=[[300, 300, 400, 400]], obj_ids=[1], update_memory=True)

MemoryBankStorage.save(
    predictor_v2,
    "versions/v2_two_objects.pkl",
    description="Added second object",
    metadata={'version': 'v2', 'author': 'user1', 'parent': 'v1'}
)

# 查看版本信息
info_v1 = MemoryBankStorage.get_info("versions/v1_initial.pkl")
info_v2 = MemoryBankStorage.get_info("versions/v2_two_objects.pkl")

print(f"V1: {info_v1['metadata']['description']}")
print(f"V2: {info_v2['metadata']['description']}")
```

### 使用範例 3: 跨設備遷移

```python
# 在 GPU 上處理並保存
predictor_gpu = SAM2DynamicInteractivePredictor(
    overrides={**overrides, 'device': 'cuda:0'},
    max_obj_num=10
)

# 處理...
for i in range(20):
    predictor_gpu(source=f"frame_{i}.jpg", bboxes=[[100, 100, 200, 200]], obj_ids=[0], update_memory=True)

# 保存
MemoryBankStorage.save(predictor_gpu, "gpu_state.pkl")

# --- 在 CPU 機器上加載 ---

predictor_cpu = SAM2DynamicInteractivePredictor(
    overrides={**overrides, 'device': 'cpu'},
    max_obj_num=10
)

# 加載（自動處理設備遷移）
MemoryBankStorage.load(predictor_cpu, "gpu_state.pkl")
# 輸出: ⚠️ 設備遷移: cuda:0 → cpu

# 繼續在 CPU 上推理
results = predictor_cpu(source="new_frame.jpg")
```

### 批量管理工具

```python
class MemoryBankManager:
    """批量管理多個 Memory Bank 存檔"""

    def __init__(self, storage_dir="memory_banks"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def list_all(self):
        """列出所有存檔"""
        pkl_files = list(self.storage_dir.glob("*.pkl"))

        print(f"\n📦 Memory Bank 存檔列表（共 {len(pkl_files)} 個）:")

        for pkl_file in sorted(pkl_files):
            try:
                info = MemoryBankStorage.get_info(pkl_file)
                metadata = info.get('metadata', {})
                config = info.get('config', {})

                print(f"\n  📄 {pkl_file.name}")
                print(f"     創建: {metadata.get('created_at', 'N/A')}")
                print(f"     Frames: {metadata.get('num_frames', 'N/A')}")
                print(f"     物件: {metadata.get('num_objects', 'N/A')}")
                print(f"     描述: {metadata.get('description', 'N/A')}")
                print(f"     大小: {pkl_file.stat().st_size / 1024 / 1024:.2f} MB")
            except Exception as e:
                print(f"\n  ❌ {pkl_file.name} (讀取失敗: {e})")

    def delete(self, filename):
        """刪除指定存檔"""
        pkl_path = self.storage_dir / filename
        json_path = pkl_path.with_suffix('.json')

        if pkl_path.exists():
            pkl_path.unlink()
            print(f"✅ 已刪除: {pkl_path}")

        if json_path.exists():
            json_path.unlink()
            print(f"✅ 已刪除: {json_path}")

    def cleanup_old(self, keep_last_n=5):
        """清理舊的存檔，只保留最新的 N 個"""
        pkl_files = sorted(
            self.storage_dir.glob("*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        to_delete = pkl_files[keep_last_n:]

        for pkl_file in to_delete:
            self.delete(pkl_file.name)

        print(f"✅ 清理完成，保留最新 {keep_last_n} 個存檔")

# 使用
manager = MemoryBankManager("my_memory_banks")
manager.list_all()
manager.cleanup_old(keep_last_n=10)
```

### 存儲最佳實踐

✅ **推薦**：

1. **定期保存**：每處理 N 個 frames 保存一次
2. **版本控制**：使用有意義的文件名和描述
3. **元數據**：添加足夠的元數據便於追蹤
4. **備份**：重要的狀態保存多份

```python
# 好的命名規範
MemoryBankStorage.save(
    predictor,
    f"states/{project_name}_frame_{num_frames:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
    description=f"Processed {num_frames} frames, tracking {len(predictor.obj_idx_set)} objects",
    metadata={
        'project': project_name,
        'stage': 'annotation',
        'quality_check': 'passed',
    }
)
```

❌ **避免**：

1. 使用非描述性的文件名（如 "state.pkl"）
2. 不添加元數據
3. 忽略設備兼容性
4. 不清理舊文件

---

## 數據保存與加載

### ⚠️ 重要：無內建保存/加載功能

SAM2DynamicInteractivePredictor **沒有內建的保存或加載方法**。這是一個**訓練無關的推理系統**，使用預訓練的 SAM2 模型權重。

### 你能保存什麼？

你可以保存**預測器狀態**（記憶體銀行和物件追蹤信息），但**不能**訓練或微調模型。

### 手動保存預測器狀態

```python
import pickle
import torch

def save_predictor_state(predictor, filepath):
    """保存預測器的記憶體狀態"""
    state = {
        'memory_bank': predictor.memory_bank,
        'obj_idx_set': predictor.obj_idx_set,
        'obj_id_to_idx': predictor.obj_id_to_idx,
        'obj_idx_to_id': predictor.obj_idx_to_id,
        'max_obj_num': predictor._max_obj_num,
        'imgsz': predictor.imgsz
    }

    with open(filepath, 'wb') as f:
        pickle.dump(state, f)

    print(f"狀態已保存到 {filepath}")

# 使用
save_predictor_state(predictor, 'predictor_state.pkl')
```

### 手動加載預測器狀態

```python
def load_predictor_state(predictor, filepath):
    """加載預測器的記憶體狀態"""
    with open(filepath, 'rb') as f:
        state = pickle.load(f)

    # 驗證配置匹配
    if predictor._max_obj_num != state['max_obj_num']:
        raise ValueError(
            f"max_obj_num 不匹配: 當前 {predictor._max_obj_num}, "
            f"保存的 {state['max_obj_num']}"
        )

    # 恢復狀態
    predictor.memory_bank = state['memory_bank']
    predictor.obj_idx_set = state['obj_idx_set']
    predictor.obj_id_to_idx = state['obj_id_to_idx']
    predictor.obj_idx_to_id = state['obj_idx_to_id']

    print(f"狀態已從 {filepath} 加載")
    print(f"已恢復 {len(predictor.memory_bank)} 個 frames")
    print(f"已追蹤物件: {predictor.obj_idx_set}")

# 使用
# 1. 創建新預測器
new_predictor = SAM2DynamicInteractivePredictor(
    overrides=overrides,
    max_obj_num=10
)

# 2. 加載保存的狀態
load_predictor_state(new_predictor, 'predictor_state.pkl')

# 3. 繼續推理
results = new_predictor(source="new_image.jpg")
```

### 保存預測結果

```python
# 保存單張圖像的預測結果
results = predictor(source="image.jpg")

# 方法 1: 使用 Ultralytics 內建保存
predictor_with_save = SAM2DynamicInteractivePredictor(
    overrides=dict(
        conf=0.01,
        task="segment",
        mode="predict",
        imgsz=1024,
        model="sam2_t.pt",
        save=True,              # 啟用保存
        project="runs/segment", # 保存目錄
        name="exp"              # 實驗名稱
    )
)

# 方法 2: 手動保存 masks
import numpy as np
from PIL import Image

results = predictor(source="image.jpg")
masks = results[0].masks.data.cpu().numpy()  # (N, H, W)

for i, mask in enumerate(masks):
    # 轉換為 0-255 範圍
    mask_img = (mask * 255).astype(np.uint8)
    Image.fromarray(mask_img).save(f"mask_{i}.png")

# 方法 3: 保存為 numpy 文件
np.save('masks.npy', masks)

# 加載
loaded_masks = np.load('masks.npy')
```

### 導出帶註釋的圖像

```python
import cv2
import numpy as np

def save_annotated_image(image_path, predictor, output_path):
    """保存帶有 mask 覆蓋層的圖像"""
    # 讀取原始圖像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 獲取預測
    results = predictor(source=image_path)
    masks = results[0].masks.data.cpu().numpy()

    # 創建顏色映射
    colors = [
        [255, 0, 0],    # 紅色
        [0, 255, 0],    # 綠色
        [0, 0, 255],    # 藍色
        [255, 255, 0],  # 黃色
        [255, 0, 255],  # 品紅色
    ]

    # 疊加 masks
    overlay = image.copy()
    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        mask_resized = cv2.resize(
            mask.astype(np.uint8),
            (image.shape[1], image.shape[0])
        )
        overlay[mask_resized > 0] = color

    # 混合
    result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

    # 保存
    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"已保存到 {output_path}")

# 使用
save_annotated_image("input.jpg", predictor, "output_annotated.jpg")
```

---

## 完整使用範例

### 範例 1: 視頻物件追蹤

```python
from ultralytics.models.sam import SAM2DynamicInteractivePredictor
import cv2
import numpy as np

# 初始化
overrides = dict(
    conf=0.01,
    task="segment",
    mode="predict",
    imgsz=1024,
    model="sam2_b.pt",
    save=False
)
predictor = SAM2DynamicInteractivePredictor(
    overrides=overrides,
    max_obj_num=5
)

# 處理視頻
video_path = "input_video.mp4"
cap = cv2.VideoCapture(video_path)

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx == 0:
        # 第一幀：手動標註物件
        # 假設我們要追蹤兩個物件
        predictor(
            source=frame,
            bboxes=[[100, 100, 300, 300], [400, 200, 600, 500]],
            obj_ids=[0, 1],
            update_memory=True
        )
    elif frame_idx == 10:
        # 第 10 幀：物件 1 的外觀改變，添加精煉提示
        predictor(
            source=frame,
            points=[[450, 350]],
            labels=[1],
            obj_ids=[1],  # 僅更新物件 1
            update_memory=True
        )
    elif frame_idx == 20:
        # 第 20 幀：出現新物件
        predictor(
            source=frame,
            bboxes=[[700, 300, 900, 600]],
            obj_ids=[2],  # 新物件
            update_memory=True
        )
    else:
        # 其他幀：僅推理
        results = predictor(source=frame)

        # 獲取 masks 並繪製
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            for i, mask in enumerate(masks):
                # 繪製到幀上
                mask_resized = cv2.resize(
                    mask,
                    (frame.shape[1], frame.shape[0])
                )
                frame[mask_resized > 0.5] = [0, 255, 0]  # 綠色覆蓋

        # 顯示結果
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

print(f"處理了 {frame_idx} 幀")
print(f"Memory bank 大小: {len(predictor.memory_bank)}")
print(f"已追蹤物件: {predictor.obj_idx_set}")
```

### 範例 2: 跨圖像物件一致性

```python
# 場景：在多張獨立圖像中分割相似物件

# 第一張圖：定義物件類別
predictor(
    source="apple1.jpg",
    bboxes=[[50, 50, 150, 150]],  # 蘋果
    obj_ids=[0],
    update_memory=True
)

# 第二張圖：使用記憶體自動檢測蘋果
results2 = predictor(source="apple2.jpg")
apple_mask_2 = results2[0].masks.data[0]  # 第一個物件（蘋果）

# 第三張圖：蘋果外觀變化（例如不同角度），添加精煉
predictor(
    source="apple3.jpg",
    points=[[200, 200]],  # 點擊蘋果
    labels=[1],
    obj_ids=[0],  # 仍然是物件 0（蘋果類別）
    update_memory=True
)

# 第四張圖：現在有更好的蘋果表示
results4 = predictor(source="apple4.jpg")

# 第五張圖：添加新類別（橙子）
predictor(
    source="orange1.jpg",
    bboxes=[[100, 100, 200, 200]],
    obj_ids=[1],  # 新物件 ID
    update_memory=True
)

# 第六張圖：同時檢測蘋果和橙子
results6 = predictor(source="fruits_mixed.jpg")
# results6[0].masks.data[0] -> 蘋果 mask
# results6[0].masks.data[1] -> 橙子 mask
```

### 範例 3: 交互式標註工具

```python
import cv2
import numpy as np

class InteractiveAnnotator:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.image_display = self.image.copy()

        # 初始化預測器
        overrides = dict(
            conf=0.01, task="segment", mode="predict",
            imgsz=1024, model="sam2_b.pt", save=False
        )
        self.predictor = SAM2DynamicInteractivePredictor(
            overrides=overrides,
            max_obj_num=10
        )

        self.points = []
        self.current_obj_id = 0
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]

        cv2.namedWindow('Annotator')
        cv2.setMouseCallback('Annotator', self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 左鍵：正向點
            self.points.append(([x, y], 1))
            cv2.circle(self.image_display, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Annotator', self.image_display)

        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右鍵：負向點
            self.points.append(([x, y], 0))
            cv2.circle(self.image_display, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('Annotator', self.image_display)

    def add_object(self):
        """使用當前點添加物件"""
        if len(self.points) == 0:
            print("沒有點！")
            return

        points = [p[0] for p in self.points]
        labels = [p[1] for p in self.points]

        # 更新記憶體
        self.predictor(
            source=self.image_path,
            points=points,
            labels=labels,
            obj_ids=[self.current_obj_id],
            update_memory=True
        )

        print(f"已添加物件 {self.current_obj_id}，包含 {len(points)} 個點")

        # 顯示結果
        self.show_results()

        # 準備下一個物件
        self.current_obj_id += 1
        self.points = []
        self.image_display = self.image.copy()

    def show_results(self):
        """顯示所有物件的 masks"""
        results = self.predictor(source=self.image_path)

        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            overlay = self.image.copy()

            for i, mask in enumerate(masks):
                color = self.colors[i % len(self.colors)]
                mask_resized = cv2.resize(
                    mask,
                    (self.image.shape[1], self.image.shape[0])
                )
                overlay[mask_resized > 0.5] = color

            result = cv2.addWeighted(self.image, 0.6, overlay, 0.4, 0)
            cv2.imshow('Results', result)

    def run(self):
        """運行交互式標註"""
        print("左鍵：正向點 | 右鍵：負向點 | SPACE：添加物件 | R：顯示結果 | Q：退出")

        cv2.imshow('Annotator', self.image_display)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):  # 空格鍵
                self.add_object()
            elif key == ord('r'):
                self.show_results()

        cv2.destroyAllWindows()

# 使用
annotator = InteractiveAnnotator("image.jpg")
annotator.run()
```

### 範例 4: 批量處理與狀態保存

```python
import os
from pathlib import Path

def batch_process_with_checkpoints(
    image_folder,
    checkpoint_folder,
    predictor
):
    """批量處理圖像並定期保存檢查點"""

    os.makedirs(checkpoint_folder, exist_ok=True)
    image_files = sorted(Path(image_folder).glob("*.jpg"))

    for i, image_path in enumerate(image_files):
        print(f"處理 {image_path.name} ({i+1}/{len(image_files)})")

        # 推理
        results = predictor(source=str(image_path))

        # 保存結果
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            np.save(
                f"{checkpoint_folder}/{image_path.stem}_masks.npy",
                masks
            )

        # 每 10 張圖像保存檢查點
        if (i + 1) % 10 == 0:
            checkpoint_path = f"{checkpoint_folder}/checkpoint_{i+1}.pkl"
            save_predictor_state(predictor, checkpoint_path)
            print(f"檢查點已保存到 {checkpoint_path}")

    # 最終檢查點
    save_predictor_state(
        predictor,
        f"{checkpoint_folder}/final_checkpoint.pkl"
    )

# 使用
predictor = SAM2DynamicInteractivePredictor(
    overrides=dict(
        conf=0.01, task="segment", mode="predict",
        imgsz=1024, model="sam2_b.pt", save=False
    ),
    max_obj_num=5
)

# 初始標註（第一張圖）
predictor(
    source="images/frame_0001.jpg",
    bboxes=[[100, 100, 300, 300]],
    obj_ids=[0],
    update_memory=True
)

# 批量處理
batch_process_with_checkpoints(
    image_folder="images",
    checkpoint_folder="checkpoints",
    predictor=predictor
)
```

---

## 重要注意事項

### 1. 性能考量

```python
# ❌ 不推薦：無限制地添加記憶體
for i in range(1000):
    predictor(
        source=f"image_{i}.jpg",
        bboxes=[[100, 100, 200, 200]],
        obj_ids=[0],
        update_memory=True  # 每次都添加到記憶體！
    )
# 結果：memory_bank 有 1000 個條目，推理會變得非常慢！

# ✅ 推薦：策略性地更新記憶體
for i in range(1000):
    if i % 50 == 0:  # 每 50 幀更新一次
        predictor(
            source=f"image_{i}.jpg",
            points=[[150, 150]],
            labels=[1],
            obj_ids=[0],
            update_memory=True
        )
    else:
        # 僅推理
        results = predictor(source=f"image_{i}.jpg")
```

**記憶體增長規則**：
- 每個 `update_memory=True` 調用 = +1 memory_bank 條目
- 更多記憶體 = 更慢的推理（線性增長）
- 推薦：每 N 幀或當物件外觀顯著改變時才更新記憶體

### 2. 物件數量限制

```python
# max_obj_num 在初始化時固定
predictor = SAM2DynamicInteractivePredictor(
    overrides=overrides,
    max_obj_num=3  # 最多 3 個物件
)

# ✅ 正確
predictor(source="img.jpg", bboxes=[[...]], obj_ids=[0], update_memory=True)
predictor(source="img.jpg", bboxes=[[...]], obj_ids=[1], update_memory=True)
predictor(source="img.jpg", bboxes=[[...]], obj_ids=[2], update_memory=True)

# ❌ 錯誤：obj_id >= max_obj_num
predictor(source="img.jpg", bboxes=[[...]], obj_ids=[3], update_memory=True)
# 拋出 AssertionError!
```

### 3. 圖像大小一致性

```python
# 初始化時設定 imgsz
predictor = SAM2DynamicInteractivePredictor(
    overrides=dict(imgsz=1024, ...),
    max_obj_num=5
)

# 所有輸入圖像都會被調整為 1024x1024
# 預測的 masks 也是 1024x1024
# 如果需要原始大小，手動調整：
import cv2
results = predictor(source="image.jpg")
mask = results[0].masks.data[0].cpu().numpy()

original_image = cv2.imread("image.jpg")
mask_resized = cv2.resize(
    mask,
    (original_image.shape[1], original_image.shape[0])
)
```

### 4. 提示組合

```python
# ❌ 不支持：單次調用中混合不同物件的提示類型
predictor(
    source="img.jpg",
    bboxes=[[100, 100, 200, 200]],  # 物件 0 的 bbox
    points=[[300, 300]],             # 物件 1 的點？
    obj_ids=[0, 1],                  # 這不會如預期工作！
    update_memory=True
)

# ✅ 正確：分別調用
predictor(
    source="img.jpg",
    bboxes=[[100, 100, 200, 200]],
    obj_ids=[0],
    update_memory=True
)
predictor(
    source="img.jpg",
    points=[[300, 300]],
    labels=[1],
    obj_ids=[1],
    update_memory=True
)
```

### 5. 分數解釋

```python
results = predictor(source="img.jpg")
scores = results[0].masks.conf  # 或從 inference() 獲取

# 分數範圍：[0, 1]
# 原始 object_score_logits: [-32, 32]
# 轉換: clamp(score / 32, min=0)
#
# 解釋：
# - > 0.5: 物件可能存在
# - > 0.0: 物件被檢測到（但低置信度）
# - 0.0: 物件不存在
```

### 6. 非重疊 Masks

```python
# 預測器默認啟用非重疊約束
predictor.non_overlap_masks  # True

# 如果有重疊的物件，後面的物件會被裁剪
# 優先級由 obj_idx 順序決定（較小的索引優先）

# 要禁用（如果需要）：
predictor.non_overlap_masks = False
# 然後重新調用 update_memory
```

### 7. 錯誤處理

```python
# 常見錯誤 1：在沒有記憶體的情況下推理
predictor = SAM2DynamicInteractivePredictor(overrides=overrides)
results = predictor(source="img.jpg")  # update_memory=False（默認）
# RuntimeError: No objects have been added to the state.

# 解決方案：首先添加至少一個物件
predictor(
    source="img.jpg",
    bboxes=[[100, 100, 200, 200]],
    obj_ids=[0],
    update_memory=True
)

# 常見錯誤 2：update_memory=True 但沒有提示
predictor(
    source="img.jpg",
    obj_ids=[0],
    update_memory=True  # 沒有 bboxes/points/masks
)
# AssertionError: bboxes, masks, or points must be provided

# 常見錯誤 3：長度不匹配
predictor(
    source="img.jpg",
    points=[[100, 100], [200, 200]],  # 2 個點
    labels=[1],                        # 1 個標籤
    obj_ids=[0],
    update_memory=True
)
# 形狀不匹配錯誤
```

---

## API 參考

### 類：SAM2DynamicInteractivePredictor

**位置**：`ultralytics/models/sam/predict.py:1669`

#### `__init__(cfg, overrides, max_obj_num=3, _callbacks=None)`

初始化預測器。

**參數**：
- `cfg` (dict): 配置字典
- `overrides` (dict): 覆蓋配置
- `max_obj_num` (int): 最大物件數量（默認: 3）
- `_callbacks` (dict): 回調函數

**屬性**：
- `memory_bank` (list): 存儲圖像狀態
- `obj_idx_set` (set): 已添加的物件索引
- `obj_id_to_idx` (OrderedDict): obj_id → obj_idx 映射
- `obj_idx_to_id` (OrderedDict): obj_idx → obj_id 映射
- `non_overlap_masks` (bool): 是否應用非重疊約束（默認: True）

---

#### `inference(im, bboxes=None, masks=None, points=None, labels=None, obj_ids=None, update_memory=False)`

主推理方法。

**參數**：
- `im` (Tensor | ndarray): 輸入圖像
- `bboxes` (list): 邊界框 `[[x1, y1, x2, y2], ...]`
- `masks` (Tensor | ndarray): 輸入 masks
- `points` (list): 點座標 `[[x, y], ...]`
- `labels` (list): 點標籤（1=正向, 0=負向）
- `obj_ids` (list): 物件 IDs
- `update_memory` (bool): 是否更新記憶體

**返回**：
- `pred_masks` (Tensor): 形狀 (C, H, W)
- `object_score_logits` (Tensor): 品質分數

**行為**：
- `update_memory=True`: 更新記憶體並返回預測
- `update_memory=False`: 僅使用現有記憶體推理

**拋出**：
- `AssertionError`: 如果 update_memory=True 但缺少 obj_ids 或提示
- `RuntimeError`: 如果沒有添加物件時嘗試推理

---

#### `update_memory(obj_ids, points=None, labels=None, masks=None)`

將圖像狀態添加到記憶體銀行。

**參數**：
- `obj_ids` (list): 物件 IDs
- `points` (Tensor): 形狀 (B, N, 2)
- `labels` (Tensor): 形狀 (B, N)
- `masks` (Tensor): 形狀 (N, H, W)

**內部流程**：
1. 為每個 obj_id 調用 `track_step()`
2. 使用 `_encode_new_memory()` 編碼 masks
3. 將 consolidated_out 附加到 memory_bank

---

#### `track_step(obj_idx=None, point=None, label=None, mask=None)`

追蹤步驟以預測 masks。

**參數**：
- `obj_idx` (int | None): 物件索引（None = 所有物件）
- `point` (Tensor): 點座標
- `label` (Tensor): 點標籤
- `mask` (Tensor): Mask 輸入

**返回**：
- dict: 包含 `pred_masks`, `pred_masks_high_res`, `obj_ptr`, `object_score_logits`

---

#### `get_im_features(img)`

提取圖像特徵。

**參數**：
- `img` (Tensor | ndarray): 輸入圖像

**設置**：
- `self.vision_feats`
- `self.vision_pos_embeds`
- `self.high_res_features`
- `self.feat_sizes`

---

#### `get_maskmem_enc()`

從記憶體銀行獲取串聯的記憶體。

**返回**：
- `memory` (Tensor): 形狀 (HW, B, C)
- `memory_pos_embed` (Tensor): 位置編碼

---

#### `_obj_id_to_idx(obj_id)`

將客戶端 obj_id 映射到模型 obj_idx。

**參數**：
- `obj_id` (int)

**返回**：
- `obj_idx` (int | None)

---

#### `_prepare_memory_conditioned_features(obj_idx)`

準備記憶體條件化的特徵。

**參數**：
- `obj_idx` (int | None)

**返回**：
- `pix_feat_with_mem` (Tensor): 形狀 (B, C, H, W)

**行為**：
- 無記憶體或初始幀：添加 no-mem 嵌入
- 有記憶體：使用 memory_attention 結合歷史特徵

---

### 調用預測器（`__call__` 方法）

預測器繼承自 `BasePredictor`，可直接調用：

```python
results = predictor(
    source="image.jpg",          # 必需
    bboxes=[[...]],              # 可選
    points=[[...]],              # 可選
    labels=[...],                # 可選
    masks=[...],                 # 可選
    obj_ids=[...],               # update_memory=True 時必需
    update_memory=False          # 默認 False
)
```

**source 支持的格式**：
- 字符串路徑：`"image.jpg"`
- numpy 數組：`np.ndarray` 形狀 (H, W, 3)
- Tensor：`torch.Tensor`
- PIL Image：`PIL.Image.Image`
- URL：`"https://..."`

**返回**：
- `Results` 對象列表（來自 Ultralytics）

**訪問結果**：
```python
results = predictor(source="img.jpg")

# Masks
masks = results[0].masks.data  # Tensor (N, H, W)
masks_np = masks.cpu().numpy()

# 分數
scores = results[0].masks.conf  # Tensor (N,)

# 邊界框（如果可用）
if results[0].boxes is not None:
    boxes = results[0].boxes.xyxy  # Tensor (N, 4)
```

---

## 總結

### 關鍵要點

1. **無顯式刪除/修改方法**：通過 `inference()` 與 `update_memory` 參數控制所有操作

2. **記憶體累積**：每次 `update_memory=True` 調用都添加新的記憶體條目

3. **Frame Cache**：
   - 同一圖像多次調用 = 多個 frames
   - memory_bank 長度 = `update_memory=True` 調用次數

4. **無訓練功能**：這是一個推理系統，使用預訓練的 SAM2 權重

5. **手動狀態管理**：需要自己實現保存/加載功能

6. **性能**：記憶體越多 = 推理越慢，需要策略性地更新

### 最佳實踐

✅ **推薦**：
- 策略性地使用 `update_memory=True`（僅在需要時）
- 每 N 幀或物件變化時更新記憶體
- 定期保存檢查點（長時間運行）
- 為不同類別使用不同的 obj_ids
- 監控 memory_bank 大小

❌ **避免**：
- 每幀都使用 `update_memory=True`
- 超過 max_obj_num 的物件
- 在沒有記憶體的情況下推理
- 忽略性能影響

### 常見工作流程

**1. 視頻追蹤**：
```
初始化 → 第一幀標註(update_memory=True) →
推理後續幀 → 必要時精煉(update_memory=True) →
新物件出現時添加(update_memory=True)
```

**2. 多圖像分割**：
```
初始化 → 定義類別(update_memory=True) →
處理類似圖像 → 外觀變化時精煉(update_memory=True)
```

**3. 交互式標註**：
```
加載圖像 → 用戶點擊/繪製 → update_memory=True →
顯示結果 → 精煉 → 保存
```

---

## 參考資源

- **源代碼**：`ultralytics/models/sam/predict.py:1669-2005`
- **文檔**：`docs/en/models/sam-2.md:229-316`
- **模型**：[Ultralytics SAM2 模型](https://docs.ultralytics.com/models/sam-2/)
- **論文**：[Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714)

---

**版本**: 1.0
**最後更新**: 2025-11-17
**作者**: Claude (Anthropic)
**適用於**: Ultralytics SAM2DynamicInteractivePredictor

如有問題或需要更多範例，請參考源代碼或提交 issue 到 [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)。
