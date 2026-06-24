package com.aegis.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

/** A flagged transaction (mirrors the inference service's snake_case NodeScore). */
public record FlagDto(@JsonProperty("node_id") int nodeId, double score, int label) {}
