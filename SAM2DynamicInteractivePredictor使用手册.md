# SAM2DynamicInteractivePredictor 使用手册

## 目录
1. [概述](#概述)
2. [核心概念](#核心概念)
3. [初始化與配置](#初始化與配置)
4. [Mask 操作詳解](#mask-操作詳解)
5. [記憶體管理與 update_memory 機制](#記憶體管理與-update_memory-機制)
6. [Frame Cache 機制詳解](#frame-cache-機制詳解)
7. [數據保存與加載](#數據保存與加載)
8. [完整使用範例](#完整使用範例)
9. [重要注意事項](#重要注意事項)
10. [API 參考](#api-參考)

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
