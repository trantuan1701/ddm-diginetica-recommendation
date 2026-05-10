# Hướng Dẫn Hiểu Repository DDM Diginetica

Tài liệu này giúp bạn hiểu repository như một project owner: biết repo làm gì, không làm gì, dữ liệu đi qua những bước nào, các bảng mart dùng để kể câu chuyện gì, và khi thuyết trình nên nói thế nào cho đúng phạm vi.

Thông điệp chính cần nhớ:

> Repo này là lớp analytics và reporting cho Data-Driven Marketing. Nhân vật chính là **SESSION**. Mô hình recommender được kế thừa qua DagsHub + MLflow, còn test examples/vocabulary có thể kế thừa từ intermediate artifacts tương thích của `recsys-group-project`. Repo này không train và không deploy SR-GNN.

## 1. Big Picture

### Repo này làm gì

Repository `ddm-diginetica-recommendation` là lớp DDM analytics/reporting cho bài toán gợi ý sản phẩm theo session trên dữ liệu Diginetica.

Nó làm các việc chính:

- Đọc dữ liệu raw Diginetica.
- Kiểm tra dữ liệu, EDA, và trực quan hóa câu chuyện vì sao nên dùng session.
- Làm sạch item views và purchases.
- Tạo các bảng processed và mart.
- Kế thừa trained recommender artifact `registered_model` từ DagsHub/MLflow registry.
- Kế thừa intermediate artifacts tương thích như `test_examples.parquet` và `item_vocab.json` từ `recsys-group-project`.
- Chạy final offline inference trong repo DDM để tạo top-20 prediction rows.
- Tính offline recommendation metrics như `HR@20`, `MRR@20`, `Catalog Coverage@20`.
- Tính marketing-safe proxy KPI như `CTR Proxy@20`, `Purchase Session Rate`, `Captured GMV Proxy@20`.
- Chuẩn bị bảng parquet để load sang PostgreSQL và Power BI.
- Hỗ trợ báo cáo cuối kỳ và dashboard.

### Repo này không làm gì

Repo này **không** (Thiếu log thật, chỉ đánh giá dựa trên train-test -> offlineoffline):

- Train SR-GNN.
- Deploy SR-GNN.
- Tạo lại train/test split.
- Chạy inference fallback nếu thiếu prediction export.
- Làm serving API, Kubernetes, monitoring.
- Tính real CTR vì không có impression logs.
- Chứng minh causal conversion uplift.
- Chứng minh ROAS.
- Tính real revenue theo kế toán.

### Quan hệ với `recsys-group-project`

Bạn có thể hiểu hai repo như sau:

```text
recsys-group-project
  -> train/chọn/promote recommender
  -> log registered_model vào DagsHub/MLflow registry
  -> lưu processed/test examples/item vocabulary theo data version

ddm-diginetica-recommendation
  -> download configured trained model artifact
  -> inherit compatible intermediate context artifacts
  -> chạy final offline inference cho test examples tương thích
  -> nối với dữ liệu marketing/session/value proxy
  -> tính metrics, KPI, mart, dashboard, report
```

`recsys-group-project` là nơi mô hình recommendation được train, chọn, promote và đóng gói artifact. Repo DDM này tiêu thụ trained model artifact đã cố định hoặc pinned, dùng intermediate artifacts tương thích để giữ đúng evaluation context, chạy inference cuối cho reporting, rồi biến output của mô hình thành phân tích marketing an toàn, có bảng, có dashboard, có câu chuyện.

### Vì sao SESSION là trung tâm

Dữ liệu Diginetica có `userId` bị thiếu rất nhiều:

- `item_views`: khoảng 69.8% thiếu `userId`.
- `purchases`: khoảng 62.8% thiếu `userId`.

Vì vậy, nếu lấy user làm trung tâm để làm RFM, CLV, churn hay segmentation thì câu chuyện sẽ yếu: nhiều hành vi không gắn được với user ổn định. Session lại có mặt trong item views, purchases, test examples và recommendation evaluation. Do đó session là đơn vị phân tích thực tế nhất.

Nói ngắn gọn khi bảo vệ project:

> Vì user identity rất thiếu, repo này không cố kể câu chuyện customer-level. Thay vào đó, project dùng session làm đơn vị chính và đánh giá khả năng bắt được next-click trong một hành trình mua sắm ngắn.

## 2. Repository Structure

```text
configs/
data/
  raw/
  processed/
  mart/
  inherited/
notebooks/
src/ddm/
sql/
reports/
docs/
```

### `configs/`

Chứa cấu hình trung tâm: [project_config.yaml](../configs/project_config.yaml).

File này định nghĩa:

- Đường dẫn raw data.
- Cấu hình DagsHub/MLflow registry: tracking URI, repo owner/name, model name, alias/version, artifact path.
- Cấu hình kế thừa artifact: `top_k`, local `inherited_root`, `context_repo_path`, và optional override cho `test_examples_path`/`item_vocab_path`.
- Đường dẫn output processed và mart.
- Safe metric framing, ví dụ `CTR Proxy@20 is Hit Rate@20...`.

Khi pipeline chạy, hầu hết đường dẫn được lấy từ file này.

### `data/raw/`

Chứa dữ liệu Diginetica gốc ở `data/raw/diginetica/`.

Các file raw:

- `train-item-views.csv`
- `train-purchases.csv`
- `products.csv`
- `product-categories.csv`
- `train-queries.csv`
- `train-clicks.csv`

### `data/processed/`

Chứa bảng đã làm sạch ở mức gần raw:

- `clean_item_views.parquet`
- `clean_purchases.parquet`

Đây là dữ liệu sau khi chuẩn hóa tên cột, parse date, bỏ dòng thiếu key quan trọng, bỏ duplicate.

### `data/mart/`

Chứa bảng phân tích cuối cùng để dùng cho PostgreSQL, Power BI và report:

- `dim_item.parquet`
- `dim_model.parquet`
- `fact_session_summary.parquet`
- `fact_purchases.parquet`
- `fact_test_examples.parquet`
- `fact_recommendations.parquet`
- `fact_recommendation_eval.parquet`
- `fact_metrics.parquet`
- `fact_marketing_kpis.parquet`
- `powerbi_notes.md`

Đây là layer quan trọng nhất cho dashboard.

### `data/inherited/`

Chứa artifact kế thừa từ DagsHub/MLflow registry:

```text
data/inherited/recsys/recsys-serving_Production_top20/
```

Bundle bắt buộc có:

- `manifest.json`
- `model_card.json` hoặc `model_card.md`
- `test_examples.parquet`
- `predictions.parquet`
- `item_vocab.json`
- `metrics.json`

Nếu thiếu `predictions.parquet`, pipeline fail rõ ràng. Repo này không tự chạy inference fallback.

### `notebooks/`

Ba notebook chính:

- `01_validate_data_and_context.ipynb`: validation, EDA, cleaning, chart PNG.
- `02_compute_metrics.ipynb`: inherited SR-GNN context, metrics, KPIs.
- `03_prepare_powerbi_tables.ipynb`: mart tables cho Power BI.

### `src/ddm/`

Code Python của pipeline:

- `io.py`: đọc config, đọc raw/mart, lưu parquet.
- `cleaning.py`: cleaning và tạo bảng session/item.
- `metrics.py`: tính recommendation metrics.
- `registry.py`: resolve/download the trained DagsHub + MLflow artifact and inherit compatible evaluation context artifacts.
- `kpis.py`: tính marketing-safe proxy KPIs.
- `pipeline.py`: runner chính cho Makefile và notebook.

### `sql/`

Chứa `schema.sql`, một PostgreSQL-friendly baseline schema cho các mart tables.

Lưu ý bảo trì: schema SQL hiện là baseline. Một số cột mới trong parquet mart như `session_length_bucket`, `item_view_count`, `item_popularity_bucket`, `target_primary_category_id` chưa được phản ánh đầy đủ trong `schema.sql`. Trước khi load sang PostgreSQL, nên đối chiếu schema parquet hiện tại với SQL.

### `reports/`

Chứa:

- `outline.md`: dàn ý báo cáo cuối kỳ.
- `figures/`: chart PNG từ EDA notebook.

## 3. Data Flow End-to-End

Luồng tổng thể:

```text
Raw Diginetica tables
  -> EDA and cleaning
  -> data/processed/
  -> data/mart/dim_item + fact_session_summary
  -> inherited context test examples + DDM-generated predictions
  -> recommendation metrics
  -> marketing proxy KPIs
  -> final mart tables
  -> PostgreSQL
  -> Power BI
  -> PDF report
```

Chi tiết hơn:

```text
train-item-views.csv
  -> clean_item_views
  -> fact_session_summary
  -> session behavior charts

train-purchases.csv + dim_item
  -> clean_purchases
  -> fact_purchases
  -> purchase/session proxy + value proxy

products.csv + product-categories.csv
  -> dim_item
  -> price_proxy + category + popularity bucket

SR-GNN inherited predictions + test_examples
  -> fact_recommendations
  -> fact_recommendation_eval
  -> fact_metrics
  -> fact_marketing_kpis
```

Điểm quan trọng: recommendation evaluation nằm ở offline test examples. Nó không phải log người dùng thật nhìn thấy recommendation.

## 4. Raw Data Understanding

### Core tables

#### `train-item-views.csv`

Header:

```text
sessionId;userId;itemId;timeframe;eventdate
```

Vai trò:

- Bảng chính để hiểu hành vi trong session.
- Mỗi dòng là một lượt xem item trong một session.
- Dùng để tạo `clean_item_views` và `fact_session_summary`.
- Dùng cho EDA session length, unique items, repeat ratio, views over time, top items/categories.

Đây là bảng quan trọng nhất cho câu chuyện session-based recommendation.

#### `train-purchases.csv`

Header:

```text
sessionId;userId;timeframe;eventdate;ordernumber;itemId
```

Vai trò:

- Mỗi dòng là một item được mua trong order/session.
- Dùng để tạo `clean_purchases`, `fact_purchases`, và purchase-related proxy KPIs.
- Có duplicate raw rows, pipeline bỏ duplicate khi cleaning.

Không dùng để claim conversion caused by recommendation. Chỉ dùng như purchase/session proxy và value proxy.

#### `products.csv`

Header:

```text
itemId;pricelog2;product.name.tokens
```

Vai trò:

- Cung cấp metadata item.
- `pricelog2` được chuyển thành `price_proxy = 2^pricelog2 - 1`.
- `price_proxy` dùng cho Captured GMV Proxy và Revenue-weighted HR.

Không được gọi là real price hoặc real revenue theo kế toán.

#### `product-categories.csv`

Header:

```text
itemId;categoryId
```

Vai trò:

- Nối item với category.
- Dùng để tạo `primary_category_id`, `category_count`, `category_ids` trong `dim_item`.
- Dùng cho chart top viewed categories và purchase value proxy by category.

### Optional/supporting tables

#### `train-queries.csv`

Header:

```text
queryId;sessionId;userId;timeframe;duration;eventdate;searchstring.tokens;categoryId;items;is.test
```

Vai trò:

- Search query events.
- Hiện chủ yếu dùng cho raw validation/profile.
- Không phải core table của mart hiện tại.

#### `train-clicks.csv`

Header:

```text
queryId;timeframe;itemId
```

Vai trò:

- Clicks liên quan query.
- Hiện chủ yếu dùng cho raw validation/profile.
- Không phải core table của mart hiện tại.

### Quy mô raw hiện tại

| Table | Rows | Columns | Duplicate rows |
|---|---:|---:|---:|
| item_views | 1,235,380 | 5 | 0 |
| clicks | 1,127,764 | 3 | 10,309 |
| queries | 923,127 | 10 | 0 |
| product_categories | 184,047 | 2 | 0 |
| products | 184,047 | 3 | 0 |
| purchases | 18,025 | 6 | 28 |

Điểm kể chuyện:

- Dataset là interaction-heavy: item views lớn hơn purchases rất nhiều.
- Purchase sparse, nên phù hợp làm proxy chứ không phải target causal.
- User ID thiếu nhiều, nên session là đơn vị đáng tin cậy hơn user.

## 5. Cleaning Logic

Cleaning nằm chủ yếu trong `src/ddm/cleaning.py`, được gọi bởi `pipeline.build_clean_layer()`.

### ID/date standardization

Hàm chính:

- `standardize_id_columns()`
- `parse_event_dates()`

Các cột raw camelCase được đổi sang snake_case:

```text
sessionId -> session_id
userId -> user_id
itemId -> item_id
categoryId -> category_id
ordernumber -> order_number
product.name.tokens -> product_name_tokens
```

Các ID được ép về nullable integer `Int64`. `eventdate` được parse thành `event_date`.

### Purchase duplicate removal

`build_clean_purchases()`:

- Parse date.
- Chuẩn hóa ID.
- Bỏ dòng thiếu `session_id`, `item_id`, `event_date`.
- Bỏ duplicate.
- Sắp xếp theo `session_id`, `event_date`, `timeframe`, `order_number`, `item_id`.

Raw purchases có 18,025 dòng, clean purchases còn 17,997 dòng.

### `dim_item` creation

`build_dim_item(products, product_categories)` tạo item dimension.

Grain:

```text
1 row per item_id
```

Nội dung chính:

- `item_id`
- `pricelog2`
- `price_proxy`
- `product_name_tokens`
- `primary_category_id`
- `category_count`
- `category_ids`

`price_proxy`:

```text
price_proxy = 2^pricelog2 - 1
```

Các giá trị non-positive hoặc non-finite được xem là missing. Hiện tại:

- `dim_item`: 184,047 items.
- Valid `price_proxy`: 126,005 items.
- Missing `price_proxy`: 58,042 items.

`add_item_popularity_features()` thêm:

- `item_view_count`
- `item_popularity_bucket`

### `clean_item_views` creation

`build_clean_item_views()` tạo bảng sạch từ `train-item-views.csv`.

Grain:

```text
1 row per cleaned item view event
```

Schema hiện tại:

| Column | Meaning |
|---|---|
| `session_id` | Session chứa view |
| `user_id` | User nếu có, có thể missing |
| `item_id` | Item được xem |
| `timeframe` | Thời gian relative trong raw |
| `event_date` | Ngày event đã parse |

Hiện có 1,235,380 rows.

### `clean_purchases` creation

Grain:

```text
1 row per cleaned purchased item in order/session
```

Schema hiện tại:

| Column | Meaning |
|---|---|
| `session_id` | Session mua |
| `user_id` | User nếu có |
| `timeframe` | Thời gian relative |
| `event_date` | Ngày purchase |
| `order_number` | Order number |
| `item_id` | Item được mua |

Hiện có 17,997 rows.

### `fact_session_summary` creation

`build_session_summary()` gom item views và purchases lên level session.

Grain:

```text
1 row per session_id
```

Các chỉ số chính:

- `view_count`
- `unique_viewed_items`
- `purchase_count`
- `unique_purchased_items`
- `order_count`
- `purchased_value_proxy`
- `has_purchase`
- `session_length_bucket`

Hiện có:

- 310,486 sessions.
- Median view count: 3.
- P90 view count: 9.
- Purchase session rate: khoảng 4.07%.

### Cleaning không làm gì

Cleaning không:

- Train model.
- Re-split train/test.
- Gán causal attribution.
- Tính real revenue.
- Tạo impression logs.

Cleaning chỉ tạo dữ liệu sạch và mart để phân tích offline.

## 6. Inherited Recommender Context

Folder:

```text
data/inherited/recsys/recsys-serving_Production_top20/
```

Folder này được tạo bởi `make inherit`. Nó kết hợp trained model artifact từ registry với intermediate context artifacts tương thích từ `recsys-group-project`. Đây là interface giữa repo DDM và recommender đã train; DDM không train lại hay re-split dữ liệu.

### `test_examples.parquet`

Grain:

```text
1 row per offline test example
```

Ý nghĩa:

- Mỗi dòng là một tình huống offline next-item prediction.
- Có prefix/session context và target item cần đo xem model có recommend đúng không.
- Đây là test split kế thừa từ registry export, repo DDM không tạo lại split.

Các cột quan trọng:

- `example_id`
- `session_id`
- `target_item_id_internal`
- `target_item_id_raw`
- `eventdate`
- `item_seq_len`
- `prefix_item_ids_internal`
- `last_item_id_internal`
- `last_item_id_raw`
- `x`, `edge_index`, `alias_inputs` cho graph/session representation.

### `predictions.parquet`

Grain:

```text
1 row per recommended item per test example
```

Ý nghĩa:

- Đây là top-20 prediction rows được DDM tạo bằng inherited trained model.
- Mỗi `example_id` có nhiều dòng, mỗi dòng là một rank recommendation.
- Cần file này để tính target rank, catalog coverage, captured value proxy và các phân tích theo session/category/value.

Các cột:

- `example_id`
- `session_id`
- `rank`
- `pred_item_id_internal`
- `pred_item_id_raw`
- `score`
- `model_key`, mặc định là `registry.model_name`.

### `item_vocab.json`

Ý nghĩa:

- Mapping giữa raw item ID và internal item ID mà recommender dùng.
- Có `item2id` và `id2item`.

Vì SR-GNN thường dùng internal item IDs, còn DDM/report dùng raw item IDs, file này là cầu nối giữa model output và item metadata.

### `metrics.json`

Hiện có:

```json
{
  "hr@k": 0.5416249183184492,
  "mrr@k": 0.18315839816184917
}
```

Ý nghĩa:

- Native aggregate metrics từ registry export.
- File này là metadata/checkpoint phụ; marts vẫn cần prediction rows để phân tích theo session, rank bucket, category, value proxy.

### Vì sao `predictions.parquet` cần thiết

`metrics.json` chỉ cho biết tổng thể HR/MRR. Nó không cho biết:

- Target item nằm ở rank mấy.
- Model recommend item nào.
- Catalog coverage thực tế.
- Hit theo session length bucket.
- Captured GMV Proxy.
- Revenue-weighted HR.
- Target category/popularity.

Vì vậy, `predictions.parquet` là file bắt buộc để biến model từ một con số aggregate thành một câu chuyện DDM có drilldown.

## 7. Metrics and KPIs

### Native recommendation metrics

#### HR@20

Intuition:

```text
HR@20 = tỷ lệ test examples mà target next item nằm trong top 20 recommendations
```

Computed in:

- `src/ddm/metrics.py`: `hit_rate_at_k()`, `score_topk_predictions()`, `evaluate_topk_predictions()`.
- Orchestrated by `src/ddm/pipeline.py`: `compute_metrics_and_kpis()`.

Stored in:

- `data/mart/fact_metrics.parquet`

Allowed claim:

- "SR-GNN bắt được held-out next item trong top 20 với tỷ lệ X trên offline test examples."
- "Offline next-click capture."

Not allowed:

- "CTR thật là X."
- "Recommendation làm người dùng click/mua nhiều hơn X."

Current values:

| Model | HR@20 |
|---|---:|
| SR-GNN | 0.541625 |
| Co-occurrence | 0.300849 |
| Popularity | 0.010216 |

#### MRR@20

Intuition:

```text
Nếu target nằm rank r trong top 20, score = 1/r.
Nếu không nằm trong top 20, score = 0.
MRR@20 = trung bình score đó.
```

Stored in:

- `fact_metrics`

Allowed claim:

- "MRR@20 đo target item xuất hiện sớm đến mức nào trong list recommendation."

Not allowed:

- "MRR chứng minh doanh thu tăng."

Current values:

| Model | MRR@20 |
|---|---:|
| SR-GNN | 0.183158 |
| Co-occurrence | 0.108070 |
| Popularity | 0.002296 |

#### Catalog Coverage@20

Intuition:

```text
Catalog Coverage@20 = unique recommended items in top 20 / catalog size
```

Stored in:

- `fact_metrics`

Allowed claim:

- "Coverage cho biết model recommend rộng hay tập trung vào vài item phổ biến."

Not allowed:

- "Coverage cao nghĩa là doanh thu cao."

Current values:

| Model | Catalog Coverage@20 |
|---|---:|
| SR-GNN | 0.935506 |
| Co-occurrence | 0.966713 |
| Popularity | 0.000867 |

### Baselines

#### Popularity baseline

Computed in:

- `src/ddm/baselines.py`: `build_popularity_baseline()`.

Logic:

- Lấy top items phổ biến từ inherited train examples tương thích với model/data version.
- Recommend cùng một danh sách top-20 cho mọi test example.

Mục đích:

- Baseline rất đơn giản để hỏi: "Model có tốt hơn chỉ recommend item phổ biến không?"

#### Co-occurrence/MBA baseline

Computed in:

- `src/ddm/baselines.py`: `build_cooccurrence_baseline()`.

Logic:

- Từ inherited train examples/session sequences, đếm transition item A -> item B.
- Với last item trong test prefix, recommend các item thường xuất hiện tiếp theo.
- Nếu thiếu transition, fallback sang popularity.

Boundary:

- Đây là classic-method offline benchmark để so sánh model với cách làm đơn giản.
- Không phải production model, không retrain SR-GNN, và không được claim causal uplift.

Mục đích:

- Baseline gần với market basket/next-item transition.
- Công bằng hơn popularity vì có dùng session context đơn giản.

### Marketing-safe proxy KPIs

Các KPI này được tính trong `src/ddm/kpis.py`, orchestrated bởi `compute_metrics_and_kpis()`, lưu ở `fact_marketing_kpis`.

#### Recommendation Success Rate@20

Formula:

```text
Recommendation Success Rate@20 = HR@20
```

Claim được phép:

- "Tỷ lệ offline next-click target được bắt trong top 20."

Không được claim:

- "Success rate ngoài thực tế."

#### CTR Proxy@20

Formula:

```text
CTR Proxy@20 = HR@20
```

Ý nghĩa:

- Đây là cách diễn đạt HR@20 theo ngôn ngữ marketing.
- Không phải CTR thật vì không có recommendation impressions.

Claim được phép:

- "CTR Proxy@20 là offline next-click capture proxy."

Không được claim:

- "CTR thật."

#### Purchase Session Rate

Formula:

```text
Purchase Session Rate = sessions with purchase / total sessions
```

Stored in:

- `fact_marketing_kpis`
- Scope: `all_sessions`

Claim được phép:

- "Khoảng 4.07% sessions có purchase trong dữ liệu."

Không được claim:

- "Recommendation gây ra purchase."

#### Hit Rate@20 among purchase sessions

Formula:

```text
Mean hit_at_k trên subset test examples thuộc purchase sessions
```

Claim được phép:

- "Trong các purchase sessions, offline next-click capture là X."

Không được claim:

- "Model tạo ra conversion."

#### Captured GMV Proxy@20

Formula:

```text
sum(target_price_proxy * hit_at_k)
```

Stored in:

- `fact_marketing_kpis`

Claim được phép:

- "Giá trị proxy của target items được model capture trong offline test."

Không được claim:

- "GMV thật."
- "Revenue thật."
- "Incremental revenue."

#### Revenue-weighted HR@20

Formula:

```text
sum(target_price_proxy * hit_at_k) / sum(target_price_proxy)
```

Ý nghĩa:

- Hit rate có trọng số theo value proxy của target item.

Claim được phép:

- "Model bắt next-click tốt hơn khi cân theo value proxy."

Không được claim:

- "Revenue uplift."

#### Captured Purchase Value Proxy@20

Formula:

```text
sum(target_price_proxy * hit_at_k)
trên target items cũng được mua trong cùng session
```

Claim được phép:

- "Offline captured purchase value proxy."

Không được claim:

- "Recommendation gây ra purchase value này."

#### Uplift/delta vs popularity baseline

Formula:

```text
(model_metric - popularity_metric) / popularity_metric
```

Stored in:

- `fact_marketing_kpis`

Claim được phép:

- "Offline relative delta so với popularity baseline."

Không được claim:

- "Causal uplift."
- "ROAS uplift."

## 8. Mart Tables and Relationships

### Current mart schemas

#### `dim_model`

Grain:

```text
1 row per model_key
```

Rows hiện tại: 3 khi train split tương thích có sẵn.

Models:

- `recsys-serving`
- `popularity_top20`
- `cooccurrence_top20`

The internal trained artifact is SR-GNN FC on `v1_strict_filter`, but the reporting join key for the inherited model is the MLflow registry model name. The other two keys are DDM-computed classic offline benchmarks.

Purpose:

- Dimension mô tả model family, role, top_k và warning text.

Connects to:

- `fact_recommendations`
- `fact_recommendation_eval`
- `fact_metrics`
- `fact_marketing_kpis`

Key:

```text
dim_model.model_key = fact_*.model_key
```

#### `dim_item`

Grain:

```text
1 row per item_id
```

Rows hiện tại: 184,047.

Columns:

- `item_id`
- `pricelog2`
- `price_proxy`
- `product_name_tokens`
- `primary_category_id`
- `category_count`
- `category_ids`
- `item_view_count`
- `item_popularity_bucket`

Purpose:

- Item metadata, category, price proxy, popularity bucket.

Connects to:

- `fact_purchases.item_id`
- `fact_test_examples.target_item_id_raw`
- `fact_recommendation_eval.target_item_id_raw`
- `fact_recommendations.pred_item_id_raw`

#### `fact_session_summary`

Grain:

```text
1 row per session_id
```

Rows hiện tại: 310,486.

Purpose:

- Session-level behavior và purchase summary.
- Bảng trung tâm cho dashboard Page 1/Page 2.

Columns chính:

- `session_id`
- `user_id`
- `first_event_date`, `last_event_date`
- `view_count`
- `unique_viewed_items`
- `purchase_count`
- `unique_purchased_items`
- `order_count`
- `purchased_value_proxy`
- `has_purchase`
- `session_length_bucket`

Connects to:

- `fact_purchases.session_id`
- `fact_test_examples.session_id`
- `fact_recommendation_eval.session_id`
- `fact_recommendations.session_id`

#### `fact_purchases`

Grain:

```text
1 row per purchased item
```

Rows hiện tại: 17,997.

Purpose:

- Purchase analysis, purchase value proxy, category purchase charts.

Columns:

- `purchase_id`
- `session_id`
- `user_id`
- `timeframe`
- `event_date`
- `order_number`
- `item_id`
- `price_proxy`
- `primary_category_id`

#### `fact_test_examples`

Grain:

```text
1 row per offline test example
```

Rows hiện tại: 45,910.

Purpose:

- Bridge giữa inherited SR-GNN test examples và DDM item/session enrichment.

Columns:

- `example_id`
- `session_id`
- `target_item_id_internal`
- `target_item_id_raw`
- `event_date`
- `target_price_proxy`
- `target_primary_category_id`
- `target_item_view_count`
- `target_item_popularity_bucket`
- `is_purchase_session`
- `session_length_bucket`

Connects to:

- `fact_recommendations.example_id`
- `fact_recommendation_eval.example_id`
- `dim_item.item_id` qua `target_item_id_raw`
- `fact_session_summary.session_id`

#### `fact_recommendations`

Grain:

```text
1 row per model_key, example_id, rank
```

Rows hiện tại: 2,754,600.

Vì:

```text
3 models * 45,910 examples * 20 ranks
```

Purpose:

- Long recommendation list cho từng model.
- Dùng để inspect item được recommend, rank, coverage.

Columns:

- `model_key`
- `example_id`
- `session_id`
- `rank`
- `pred_item_id_internal`
- `pred_item_id_raw`
- `score`

#### `fact_recommendation_eval`

Grain:

```text
1 row per model_key, example_id
```

Rows hiện tại: 137,730.

Vì:

```text
3 models * 45,910 examples
```

Purpose:

- Bảng đánh giá per-example.
- Dùng tốt cho Power BI slicing theo model, session length, purchase session, target category, target popularity, value proxy.

Columns:

- `model_key`
- `example_id`
- `session_id`
- `target_item_id_raw`
- `hit_at_k`
- `target_rank`
- `reciprocal_rank`
- `target_price_proxy`
- `target_primary_category_id`
- `target_item_view_count`
- `target_item_popularity_bucket`
- `captured_value_proxy`
- `is_purchase_session`
- `session_length_bucket`

#### `fact_metrics`

Grain:

```text
1 row per model_key, metric_name
```

Rows hiện tại: 9.

Purpose:

- Summary recommendation metrics.
- Dùng cho chart model comparison.

Metrics:

- `HR@20`
- `MRR@20`
- `Catalog Coverage@20`

#### `fact_marketing_kpis`

Grain:

```text
1 row per model_key, kpi_name
```

Rows hiện tại: 25.

Purpose:

- Business-friendly proxy KPIs.
- Dùng cho dashboard executive overview và marketing-safe story.

Important fields:

- `kpi_name`
- `kpi_value`
- `kpi_scope`
- `warning_text`

### Relationships in words

```text
dim_model
  -> fact_recommendations
  -> fact_recommendation_eval
  -> fact_metrics
  -> fact_marketing_kpis
```

```text
dim_item
  -> fact_purchases via item_id
  -> fact_test_examples via target_item_id_raw
  -> fact_recommendation_eval via target_item_id_raw
  -> fact_recommendations via pred_item_id_raw
```

```text
fact_session_summary
  -> fact_purchases via session_id
  -> fact_test_examples via session_id
  -> fact_recommendation_eval via session_id
  -> fact_recommendations via session_id
```

```text
fact_test_examples
  -> fact_recommendations via example_id
  -> fact_recommendation_eval via example_id
```

Power BI recommendation:

- Use `dim_model` as one-to-many to model facts.
- Use `dim_item` carefully because it may connect to both target item and predicted item. In Power BI, you may need role-playing relationships or one active relationship plus inactive alternatives.
- Use `fact_session_summary` as session dimension/fact hybrid for session slicing.

## 9. Notebooks

### `01_validate_data_and_context.ipynb`

Purpose:

- Validate raw data.
- Build session-centered EDA story.
- Save useful charts to `reports/figures/`.
- Build and save first cleaned outputs.

Inputs:

- `configs/project_config.yaml`
- Raw Diginetica files in `data/raw/diginetica/`

Main actions:

- Raw table overview.
- Duplicate check.
- Missing `userId` rate.
- Date parsing checks.
- Item/product/category coverage.
- Build `clean_item_views`, `clean_purchases`, `dim_item`, `fact_session_summary`.
- Chart session behavior.
- Chart purchase/value proxy.
- If mart files exist, chart recommendation metrics.

Outputs:

- `data/processed/clean_item_views.parquet`
- `data/processed/clean_purchases.parquet`
- `data/mart/dim_item.parquet`
- `data/mart/fact_session_summary.parquet`
- PNG charts in `reports/figures/`

What to check:

- Notebook JSON runs without error.
- Figures exist.
- Missing `userId` chart supports session framing.
- `price_proxy` has no non-positive valid values.
- `session_length_bucket` exists.

### `02_compute_metrics.ipynb`

Purpose:

- Prepare inherited SR-GNN context.
- Validate inherited schemas.
- Compute model metrics, recommendation rows from final inference, and marketing-safe KPIs.

Inputs:

- `data/inherited/recsys/...`
- `data/mart/dim_item.parquet`
- `data/mart/fact_session_summary.parquet`
- `data/processed/clean_purchases.parquet`

Main actions:

- Calls `inherit_recsys_context()` when downloading is needed.
- Checks `test_examples.parquet`.
- Calls `compute_metrics_and_kpis()`.
- Shows `fact_metrics`, `fact_marketing_kpis`, `fact_recommendations`, `fact_recommendation_eval`.

Outputs:

- `fact_metrics.parquet`
- `fact_marketing_kpis.parquet`
- `fact_recommendations.parquet`
- `fact_test_examples.parquet`
- `fact_recommendation_eval.parquet`
- `dim_model.parquet`

What to check:

- SR-GNN predictions are available.
- `fact_recommendations` has 20 rows per example per model.
- `fact_metrics` contains metrics from the inherited model's final inference rows.
- KPI wording has warning text.

### `03_prepare_powerbi_tables.ipynb`

Purpose:

- Prepare final mart tables for PostgreSQL/Power BI.

Inputs:

- Config.
- Processed and mart outputs.
- Inherited/context outputs.

Main actions:

- Calls `prepare_powerbi_marts()`.
- Recomputes/refreshes metrics marts.
- Creates `fact_purchases`.
- Prints required table previews.
- Prints Power BI notes.

Outputs:

- Final mart parquet files in `data/mart/`.
- `data/mart/powerbi_notes.md`.

What to check:

- All required mart tables exist.
- Table row counts look expected.
- Power BI notes include limitations.
- PostgreSQL import ownership is handled separately; check schema alignment with current parquet columns before loading.

## 10. Source Code Modules

### `io.py`

Problem it solves:

- Centralized file I/O.

Important functions:

- `load_config()`: reads YAML config.
- `read_table()`: reads CSV/parquet/JSON.
- `load_raw_tables()`: loads all raw tables declared in config.
- `save_parquet()`: writes parquet with parent directory creation.
- `read_json()`: reads JSON object.

Connects to:

- All notebooks.
- `pipeline.py`.

### `cleaning.py`

Problem it solves:

- Turns raw Diginetica data into clean session/item/purchase tables.

Important functions:

- `standardize_id_columns()`
- `parse_event_dates()`
- `build_dim_item()`
- `add_item_popularity_features()`
- `build_clean_item_views()`
- `build_clean_purchases()`
- `build_session_summary()`

Connects to:

- `make validate`.
- Notebook 01.
- `pipeline.build_clean_layer()`.

### `metrics.py`

Problem it solves:

- Compute offline next-item recommendation metrics from top-k recommendation rows.

Important functions:

- `hit_rate_at_k()`
- `mrr_at_k()`
- `target_rank_at_k()`
- `catalog_coverage_at_k()`
- `score_topk_predictions()`
- `evaluate_topk_predictions()`

Connects to:

- `pipeline.compute_metrics_and_kpis()`.
- `fact_metrics`.
- `fact_recommendation_eval`.

### `registry.py`

Problem it solves:

- Resolve and download the configured MLflow registered model export.
- Validate the inherited bundle before any metrics or marts are computed.

Important functions:

- `tracking_uri_from_config()`
- `bundle_directory()`
- `resolve_registry_reference()`
- `download_registry_bundle()`
- `validate_inherited_bundle()`

Connects to:

- `make inherit`.
- `pipeline.compute_metrics_and_kpis()`.
- Local `data/inherited/recsys/...` bundle.

### `kpis.py`

Problem it solves:

- Convert recommendation evaluation into marketing-safe proxy KPIs.

Important functions:

- `price_from_pricelog2()`
- `ctr_proxy_from_hit_rate()`
- `purchase_session_rate()`
- `revenue_weighted_hit_rate()`
- `enrich_scored_examples_for_value()`
- `compute_model_proxy_kpis()`
- `native_metric_proxy_kpis()`

Connects to:

- `fact_marketing_kpis`.
- `fact_recommendation_eval`.
- Notebook 02.

### `pipeline.py`

Problem it solves:

- Orchestrates the whole repo workflow.

Important functions:

- `build_clean_layer()`: used by `make validate`.
- `inherit_recsys_context()`: downloads the configured model artifact, inherits compatible intermediate context, and validates the local bundle.
- `compute_metrics_and_kpis()`: used by `make metrics`.
- `prepare_powerbi_marts()`: used by `make marts`.
- `main()`: CLI entrypoint for Makefile.

Connects to:

- Makefile.
- All notebooks.
- All output marts.

Important behavior:

- If `predictions.parquet` exists, SR-GNN is evaluated from prediction rows.
- If not, native aggregate metrics may still be used, but detailed DDM slicing is limited.
- Final prediction rows are computed from the inherited trained model.

## 11. Makefile Workflow

Makefile commands:

```bash
make validate
make metrics
make marts
```

Recommended order:

```text
1. make validate
2. make metrics
3. make marts
```

### `make validate`

Runs:

```bash
PYTHONPATH=src python -m ddm.pipeline validate
```

Produces:

- `data/processed/clean_item_views.parquet`
- `data/processed/clean_purchases.parquet`
- `data/mart/dim_item.parquet`
- `data/mart/fact_session_summary.parquet`

Use when:

- Raw data changed.
- Cleaning logic changed.
- You need refreshed session summary.

### `make metrics`

Runs:

```bash
PYTHONPATH=src python -m ddm.pipeline metrics
```

Produces:

- `fact_metrics`
- `fact_marketing_kpis`
- `fact_recommendations`
- `fact_test_examples`
- `fact_recommendation_eval`
- `dim_model`

Use when:

- Inherited predictions/test examples changed.
- You need updated model comparison and KPIs.

### `make marts`

Runs:

```bash
PYTHONPATH=src python -m ddm.pipeline marts
```

Produces/refreshes:

- All final mart tables.
- `fact_purchases`.
- `powerbi_notes.md`.

Use when:

- Preparing PostgreSQL/Power BI export.
- Finalizing dashboard inputs.

## 12. Power BI and Reporting Story

### Page 1: Executive Overview

Goal:

- Show what project is about and why session matters.

Suggested visuals:

- KPI cards: total sessions, total item views, purchase session rate, number of models.
- Bar chart: row count by raw table.
- Bar chart: missing `userId` rate.
- Short note: "Session is central because user identity is sparse."

Safe wording:

- "Offline session-based recommendation analytics."
- "Purchase/session proxy."

Avoid:

- "Real CTR."
- "Revenue uplift."

### Page 2: Session Behavior

Goal:

- Prove that session is a meaningful behavior unit.

Suggested visuals:

- Histogram: session length.
- Histogram: unique items per session.
- Histogram: repeat ratio.
- Line chart: item views over time.
- Bar chart: top viewed categories.
- Bar chart: purchase rate by session length bucket.

Talking point:

> Sessions are short but contain sequential item transitions. That makes next-item recommendation a reasonable offline task.

### Page 3: Model vs Classic Benchmarks

Goal:

- Compare the inherited registered model against popularity and co-occurrence classic benchmarks.

Suggested visuals:

- Bar chart: HR@20 by model.
- Bar chart: MRR@20 by model.
- Bar chart: Catalog Coverage@20 by model.
- Target rank bucket chart.
- HR@20 by session length bucket and model.

Talking point:

> The recommender is inherited from the configured MLflow artifact, with compatible intermediate evaluation context inherited from `recsys-group-project`. This repo runs final offline inference and evaluates it in a DDM-safe way.

### Page 4: Marketing Proxy KPIs and Limitations

Goal:

- Translate offline metrics into marketing language without overclaiming.

Suggested visuals:

- Recommendation Success Rate@20.
- CTR Proxy@20.
- Captured GMV Proxy@20.
- Revenue-weighted HR@20.
- Captured Purchase Value Proxy@20.
- Warning text table from `fact_marketing_kpis`.

Talking point:

> These are proxy KPIs. They help discuss business relevance, but they are not real CTR, not causal conversion, and not audited revenue.

## 13. Limitations

Be very explicit in report and presentation.

### No recommendation impressions

Dataset does not show:

- Which recommendation list was actually shown to user.
- Whether user saw recommendation.
- Position/exposure logs in production.

So real CTR cannot be computed.

### No real CTR

`CTR Proxy@20` is:

```text
CTR Proxy@20 = HR@20
```

It means offline next-click capture, not actual click-through rate.

### No causal conversion uplift

There is no randomized experiment, no treatment/control group, and no exposure assignment. Therefore:

- Do not say recommendation caused purchase.
- Do not say conversion uplift.

### No ROAS

There is no ad spend, campaign cost, or causal revenue attribution. Therefore:

- Do not say ROAS.
- Do not say marketing ROI.

### `price_proxy` is not accounting revenue

`price_proxy` comes from:

```text
2^pricelog2 - 1
```

It is useful for value-oriented offline weighting, but not audited price or revenue.

### Offline evaluation only

All recommendation metrics are evaluated on held-out offline test examples. They are useful for model comparison, not proof of real-world business impact.

## 14. How To Explain This Repo In 2 Minutes

Bạn có thể nói như sau:

> Project này là lớp Data-Driven Marketing analytics cho bài toán recommendation trên dữ liệu Diginetica. Điểm quan trọng là em không lấy user làm trung tâm, vì `userId` bị thiếu rất nhiều, khoảng 70% ở item views. Vì vậy, đơn vị phân tích chính là session.
>
> Repo này không train SR-GNN. Recommender được kế thừa từ DagsHub/MLflow artifact `registered_model`, còn test examples và vocabulary có thể kế thừa từ intermediate artifacts tương thích trong `recsys-group-project`. Repo DDM này dùng các artifact đó để chạy final offline inference, tạo prediction rows, rồi nối với dữ liệu item, category, purchase và price proxy để tạo bảng phân tích.
>
> Pipeline bắt đầu từ raw Diginetica tables, làm sạch item views và purchases, tạo `dim_item` và `fact_session_summary`. Sau đó repo dùng prediction rows và test examples từ inherited artifacts để tính HR@20, MRR@20, Catalog Coverage@20. Repo cũng tính popularity và co-occurrence baselines từ train split tương thích để so sánh model với classic methods.
>
> Về marketing, repo chỉ dùng proxy KPIs an toàn. `CTR Proxy@20` thực chất là HR@20, tức offline next-click capture, không phải CTR thật. `Captured GMV Proxy` và `Revenue-weighted HR` dùng `price_proxy` từ `pricelog2`, không phải doanh thu thật. Vì không có impression logs hay A/B test, project không claim conversion uplift, ROAS hay causal revenue.
>
> Output cuối cùng là các mart tables để load vào PostgreSQL và Power BI, giúp trình bày câu chuyện: session là trung tâm, SR-GNN bắt next-click trong offline evaluation, và các KPI marketing chỉ là proxy có giới hạn rõ ràng.

## 15. Next Steps Checklist

### EDA and figures

- Kiểm tra `reports/figures/` có đủ chart PNG.
- Đảm bảo chart missing `userId` được dùng để giải thích vì sao session là trung tâm.
- Đảm bảo chart session length/unique items/repeat ratio hỗ trợ câu chuyện next-item recommendation.

### PostgreSQL

- PostgreSQL schema/import work is a separate handoff from this pipeline refactor.
- Collaborator phụ trách SQL nên đối chiếu `sql/schema.sql` với schema parquet hiện tại trước khi import.
- Load các mart parquet vào PostgreSQL.
- Kiểm tra row counts sau khi load.

### Power BI relationships

- Tạo relationship từ `dim_model.model_key` sang các fact model tables.
- Tạo relationship từ `fact_session_summary.session_id` sang session-related facts.
- Tạo relationship từ `fact_test_examples.example_id` sang recommendation facts.
- Xử lý `dim_item` cẩn thận nếu có cả target item và predicted item.

### Dashboard

- Page 1: Executive overview.
- Page 2: Session behavior.
- Page 3: SR-GNN offline inference metrics.
- Page 4: Marketing proxy KPIs and limitations.

### PDF report

- Dùng `reports/outline.md` làm khung.
- Mở đầu bằng vấn đề user identity missing.
- Trình bày session-centered data flow.
- Nêu SR-GNN là inherited backbone.
- So sánh model bằng offline metrics.
- Kết thúc bằng limitations và safe claims.

### Wording check

Trước khi nộp, rà lại các từ sau:

Allowed:

- CTR Proxy
- purchase/session proxy
- value proxy
- offline next-click capture
- offline recommendation evaluation

Not allowed:

- real CTR
- real revenue
- causal uplift
- conversion caused by recommendation
- ROAS
- incremental revenue

## Quick Maintenance Notes

- Nếu raw data thay đổi: chạy `make validate`.
- Nếu inherited predictions/test examples thay đổi: chạy `make metrics`.
- Nếu chuẩn bị Power BI: chạy `make marts`.
- Nếu chỉ cần hiểu dữ liệu và thuyết trình: đọc README, notebook 01, `fact_metrics`, `fact_marketing_kpis`, và tài liệu này.
- Nếu có lỗi schema khi load PostgreSQL: kiểm tra khác biệt giữa `sql/schema.sql` và parquet schemas trong `data/mart/`.
