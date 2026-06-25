package com.aegis.service;

import com.aegis.client.InferenceClient;
import com.aegis.dto.FlagDto;
import com.aegis.dto.MetricsDto;
import com.aegis.entity.CachedExplanation;
import com.aegis.repository.CachedExplanationRepository;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.List;
import org.springframework.stereotype.Service;

/** Orchestrates the inference service for flags / explanations / metrics / adversarial, applying
 *  graph-capping and caching explanations in Postgres. */
@Service
public class AnalysisService {

    private final InferenceClient inference;
    private final GraphCappingService capping;
    private final CachedExplanationRepository explanationCache;

    public AnalysisService(InferenceClient inference, GraphCappingService capping,
                           CachedExplanationRepository explanationCache) {
        this.inference = inference;
        this.capping = capping;
        this.explanationCache = explanationCache;
    }

    public List<FlagDto> flags(double threshold, int limit) { return inference.flags(threshold, limit); }

    public MetricsDto metrics(String split) { return inference.metrics(split); }

    /** Async LLM narration for a node ({ready, summary}); the UI polls this to upgrade the
     *  instant template summary. Stateless passthrough — the inference service caches it. */
    public JsonNode summary(int nodeId) { return inference.summary(nodeId); }

    public JsonNode adversarial() { return inference.adversarial(); }

    /** Explanation contract for a node, capped for rendering, cached by (dataset, node). */
    public JsonNode explain(Long datasetId, int nodeId) {
        return explanationCache.findByDatasetIdAndNodeId(datasetId, nodeId)
                .map(CachedExplanation::getPayload)
                .orElseGet(() -> {
                    JsonNode capped = capping.capExplanation(inference.explain(nodeId, "auto", 2, 400));
                    explanationCache.save(new CachedExplanation(datasetId, nodeId, capped));
                    return capped;
                });
    }
}
