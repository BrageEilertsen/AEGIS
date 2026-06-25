package com.aegis.client;

import com.aegis.dto.FlagDto;
import com.aegis.dto.MetricsDto;
import com.aegis.exception.InferenceUnavailableException;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.List;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import org.springframework.stereotype.Component;

/** Thin typed client over the FastAPI inference service. Explanation/adversarial payloads are
 *  returned as raw JsonNode (the Python side owns the versioned contract; we forward it verbatim). */
@Component
public class InferenceClient {

    private final RestClient client;

    public InferenceClient(RestClient inferenceRestClient) {
        this.client = inferenceRestClient;
    }

    public JsonNode info() {
        return get("/health", JsonNode.class);
    }

    public List<FlagDto> flags(double threshold, int limit) {
        return get("/flags?threshold=%s&limit=%d".formatted(threshold, limit),
                   new org.springframework.core.ParameterizedTypeReference<List<FlagDto>>() {});
    }

    public MetricsDto metrics(String split) {
        return get("/metrics?split=" + split, MetricsDto.class);
    }

    public JsonNode explain(int nodeId, String method, int numHops, int maxNodes) {
        try {
            return client.post().uri("/explain")
                    .body(java.util.Map.of("node_id", nodeId, "method", method,
                                           "num_hops", numHops, "max_nodes", maxNodes))
                    .retrieve().body(JsonNode.class);
        } catch (RestClientException e) {
            throw new InferenceUnavailableException("explain failed for node " + nodeId, e);
        }
    }

    /** Poll the async LLM narration for a node: {ready, summary}. */
    public JsonNode summary(int nodeId) {
        return get("/summary/" + nodeId, JsonNode.class);
    }

    public JsonNode adversarial() {
        try {
            return client.post().uri("/adversarial").body(java.util.Map.of()).retrieve().body(JsonNode.class);
        } catch (RestClientException e) {
            throw new InferenceUnavailableException("adversarial artifact unavailable", e);
        }
    }

    private <T> T get(String uri, Class<T> type) {
        try { return client.get().uri(uri).retrieve().body(type); }
        catch (RestClientException e) { throw new InferenceUnavailableException("GET " + uri + " failed", e); }
    }

    private <T> T get(String uri, org.springframework.core.ParameterizedTypeReference<T> type) {
        try { return client.get().uri(uri).retrieve().body(type); }
        catch (RestClientException e) { throw new InferenceUnavailableException("GET " + uri + " failed", e); }
    }
}
