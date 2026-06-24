package com.aegis.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

/** Held-out metrics for the illicit class (accuracy is intentionally absent — spec §7.6). */
public record MetricsDto(
        @JsonProperty("pr_auc") Double prAuc,
        @JsonProperty("roc_auc") Double rocAuc,
        @JsonProperty("recall_at_precision") Double recallAtPrecision,
        @JsonProperty("min_precision") Double minPrecision,
        @JsonProperty("f1_illicit") Double f1Illicit,
        @JsonProperty("n_total") Integer nTotal,
        @JsonProperty("n_pos") Integer nPos,
        @JsonProperty("confusion_matrix") java.util.Map<String, Integer> confusionMatrix) {}
