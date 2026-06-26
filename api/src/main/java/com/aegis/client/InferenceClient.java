package com.aegis.client;

import com.aegis.dto.FlagDto;
import com.aegis.dto.MetricsDto;
import com.aegis.exception.InferenceUnavailableException;
import io.github.resilience4j.bulkhead.annotation.Bulkhead;
import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import io.github.resilience4j.retry.annotation.Retry;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;

/** Thin typed client over the FastAPI inference service. Explanation/adversarial payloads are
 *  returned as raw JsonNode (the Python side owns the versioned contract; we forward it verbatim).
 *
 *  Every call is guarded by Resilience4j: a {@code bulkhead} caps concurrent in-flight calls, a
 *  {@code retry} rides out transient blips, and a {@code circuitbreaker} trips after repeated
 *  failures so a slow/down model fails fast (via the per-method fallback) instead of tying up
 *  request threads and cascading. The inference service is the one external dependency here, so
 *  isolating it is the whole resilience story. */
@Component
public class InferenceClient {

    private static final String INF = "inference";
    private static final Logger log = LoggerFactory.getLogger(InferenceClient.class);

    private final RestClient client;

    public InferenceClient(RestClient inferenceRestClient) {
        this.client = inferenceRestClient;
    }

    @Bulkhead(name = INF)
    @CircuitBreaker(name = INF, fallbackMethod = "infoFallback")
    @Retry(name = INF)
    public com.fasterxml.jackson.databind.JsonNode info() {
        return get("/health", com.fasterxml.jackson.databind.JsonNode.class);
    }

    @Bulkhead(name = INF)
    @CircuitBreaker(name = INF, fallbackMethod = "flagsFallback")
    @Retry(name = INF)
    public List<FlagDto> flags(double threshold, int limit) {
        return get("/flags?threshold=%s&limit=%d".formatted(threshold, limit),
                   new ParameterizedTypeReference<List<FlagDto>>() {});
    }

    @Bulkhead(name = INF)
    @CircuitBreaker(name = INF, fallbackMethod = "metricsFallback")
    @Retry(name = INF)
    public MetricsDto metrics(String split) {
        return get("/metrics?split=" + split, MetricsDto.class);
    }

    @Bulkhead(name = INF)
    @CircuitBreaker(name = INF, fallbackMethod = "jsonFallback")
    @Retry(name = INF)
    public com.fasterxml.jackson.databind.JsonNode explain(int nodeId, String method, int numHops, int maxNodes) {
        try {
            return client.post().uri("/explain")
                    .body(java.util.Map.of("node_id", nodeId, "method", method,
                                           "num_hops", numHops, "max_nodes", maxNodes))
                    .retrieve().body(com.fasterxml.jackson.databind.JsonNode.class);
        } catch (RestClientException e) {
            throw new InferenceUnavailableException("explain failed for node " + nodeId, e);
        }
    }

    /** Poll the async LLM narration for a node: {ready, summary}. */
    @Bulkhead(name = INF)
    @CircuitBreaker(name = INF, fallbackMethod = "jsonFallback")
    @Retry(name = INF)
    public com.fasterxml.jackson.databind.JsonNode summary(int nodeId) {
        return get("/summary/" + nodeId, com.fasterxml.jackson.databind.JsonNode.class);
    }

    @Bulkhead(name = INF)
    @CircuitBreaker(name = INF, fallbackMethod = "adversarialFallback")
    @Retry(name = INF)
    public com.fasterxml.jackson.databind.JsonNode adversarial() {
        try {
            return client.post().uri("/adversarial").body(java.util.Map.of())
                    .retrieve().body(com.fasterxml.jackson.databind.JsonNode.class);
        } catch (RestClientException e) {
            throw new InferenceUnavailableException("adversarial artifact unavailable", e);
        }
    }

    // --- Resilience4j fallbacks (invoked on bulkhead-full, open circuit, or exhausted retries) ---
    // Signature = original args + the Throwable; they surface a clean 502 instead of a raw error.
    private com.fasterxml.jackson.databind.JsonNode infoFallback(Throwable ex) { throw unavailable("/health", ex); }
    private List<FlagDto> flagsFallback(double t, int l, Throwable ex) { throw unavailable("flags", ex); }
    private MetricsDto metricsFallback(String s, Throwable ex) { throw unavailable("metrics", ex); }
    private com.fasterxml.jackson.databind.JsonNode jsonFallback(int nodeId, String m, int h, int mn, Throwable ex) {
        throw unavailable("explain node " + nodeId, ex);
    }
    private com.fasterxml.jackson.databind.JsonNode jsonFallback(int nodeId, Throwable ex) {
        throw unavailable("summary node " + nodeId, ex);
    }
    private com.fasterxml.jackson.databind.JsonNode adversarialFallback(Throwable ex) { throw unavailable("adversarial", ex); }

    private InferenceUnavailableException unavailable(String what, Throwable ex) {
        if (ex instanceof InferenceUnavailableException iue) {
            return iue;   // already the right type (e.g. from explain/adversarial) — don't double-wrap
        }
        log.warn("Inference call '{}' failed/short-circuited: {}", what, ex.toString());
        return new InferenceUnavailableException("inference unavailable for " + what, ex);
    }

    private <T> T get(String uri, Class<T> type) {
        try { return client.get().uri(uri).retrieve().body(type); }
        catch (RestClientException e) { throw new InferenceUnavailableException("GET " + uri + " failed", e); }
    }

    private <T> T get(String uri, ParameterizedTypeReference<T> type) {
        try { return client.get().uri(uri).retrieve().body(type); }
        catch (RestClientException e) { throw new InferenceUnavailableException("GET " + uri + " failed", e); }
    }
}
