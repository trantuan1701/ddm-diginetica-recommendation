let
    // ---------- Helpers ----------
    NormalizeColumns = (tbl as table) as table =>
        let
            renamed = Table.RenameColumns(
                tbl,
                {
                    {"sessionId", "session_id"},
                    {"itemId", "item_id"},
                    {"eventdate", "event_date"},
                    {"price_log2", "pricelog2"}
                },
                MissingField.Ignore
            )
        in
            renamed,

    ToNumberColumns = (tbl as table, cols as list) as table =>
        Table.TransformColumns(
            tbl,
            List.Transform(cols, (c) => {c, each try Number.From(_) otherwise null, type number}),
            null,
            MissingField.Ignore
        ),

    ToTextColumns = (tbl as table, cols as list) as table =>
        Table.TransformColumns(
            tbl,
            List.Transform(cols, (c) => {c, each try Text.From(_) otherwise null, type text}),
            null,
            MissingField.Ignore
        ),

    // ---------- Load core marts ----------
    FactRecommendationEvalRaw = Parquet.Document(File.Contents("data/mart/fact_recommendation_eval.parquet")),
    FactMetricsRaw = Parquet.Document(File.Contents("data/mart/fact_metrics.parquet")),
    FactMarketingKpisRaw = Parquet.Document(File.Contents("data/mart/fact_marketing_kpis.parquet")),
    DimModelRaw = Parquet.Document(File.Contents("data/mart/dim_model.parquet")),
    DimItemRaw = Parquet.Document(File.Contents("data/mart/dim_item.parquet")),

    FactRecommendationEval0 = NormalizeColumns(FactRecommendationEvalRaw),
    FactMetrics0 = NormalizeColumns(FactMetricsRaw),
    FactMarketingKpis0 = NormalizeColumns(FactMarketingKpisRaw),
    DimModel0 = NormalizeColumns(DimModelRaw),
    DimItem0 = NormalizeColumns(DimItemRaw),

    FactRecommendationEval = ToNumberColumns(
        FactRecommendationEval0,
        {"session_id", "item_id", "target_item_id_raw", "hit_at_k", "rank", "target_rank", "reciprocal_rank", "captured_value_proxy"}
    ),
    FactMetrics = ToNumberColumns(FactMetrics0, {"metric_value", "k"}),
    FactMarketingKpis = ToNumberColumns(FactMarketingKpis0, {"kpi_value", "k"}),
    DimModel = ToTextColumns(DimModel0, {"model_key", "model_label"}),
    DimItem = ToNumberColumns(DimItem0, {"item_id", "category_id", "pricelog2", "price_proxy"}),

    // ---------- Load raw CSVs ----------
    ItemViewsCsv = Csv.Document(
        File.Contents("data/raw/diginetica/train-item-views.csv"),
        [Delimiter = ";", Encoding = 65001, QuoteStyle = QuoteStyle.None]
    ),
    ItemViewsHeaders = Table.PromoteHeaders(ItemViewsCsv, [PromoteAllScalars = true]),
    ItemViews1 = NormalizeColumns(ItemViewsHeaders),
    ItemViews2 = ToNumberColumns(ItemViews1, {"session_id", "item_id", "timeframe"}),
    ItemViews = Table.TransformColumnTypes(ItemViews2, {{"event_date", type datetime}}, "en-US"),

    PurchasesCsv = Csv.Document(
        File.Contents("data/raw/diginetica/train-purchases.csv"),
        [Delimiter = ";", Encoding = 65001, QuoteStyle = QuoteStyle.None]
    ),
    PurchasesHeaders = Table.PromoteHeaders(PurchasesCsv, [PromoteAllScalars = true]),
    Purchases1 = NormalizeColumns(PurchasesHeaders),
    Purchases2 = ToNumberColumns(Purchases1, {"session_id", "item_id", "timeframe", "quantity"}),
    Purchases = Table.Distinct(Purchases2, {"session_id", "item_id", "timeframe"}),

    // ---------- Optional helper tables ----------
    SessionSummaryCsv = Csv.Document(
        File.Contents("dashboards/powerbi_data/pbi_session_summary.csv"),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]
    ),
    SessionSummaryHeaders = Table.PromoteHeaders(SessionSummaryCsv, [PromoteAllScalars = true]),
    SessionSummary0 = NormalizeColumns(SessionSummaryHeaders),
    SessionSummary = ToNumberColumns(SessionSummary0, {"session_id", "view_count", "unique_items_viewed", "purchase_count", "quantity_sum"}),

    ModelMetricsSummaryCsv = Csv.Document(
        File.Contents("dashboards/powerbi_data/pbi_model_metrics_summary.csv"),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]
    ),
    ModelMetricsSummaryHeaders = Table.PromoteHeaders(ModelMetricsSummaryCsv, [PromoteAllScalars = true]),
    ModelMetricsSummary0 = NormalizeColumns(ModelMetricsSummaryHeaders),
    ModelMetricsSummary = ToNumberColumns(ModelMetricsSummary0, {"HR@20", "MRR@20", "Coverage@20"}),

    DataQualitySummaryCsv = Csv.Document(
        File.Contents("dashboards/powerbi_data/pbi_data_quality_summary.csv"),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]
    ),
    DataQualitySummary = Table.PromoteHeaders(DataQualitySummaryCsv, [PromoteAllScalars = true])
in
    [
        FactRecommendationEval = FactRecommendationEval,
        FactMetrics = FactMetrics,
        FactMarketingKpis = FactMarketingKpis,
        DimModel = DimModel,
        DimItem = DimItem,
        ItemViews = ItemViews,
        Purchases = Purchases,
        SessionSummary = SessionSummary,
        ModelMetricsSummary = ModelMetricsSummary,
        DataQualitySummary = DataQualitySummary
    ]
