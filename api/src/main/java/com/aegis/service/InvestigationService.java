package com.aegis.service;

import com.aegis.exception.InferenceUnavailableException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.aegis.client.InferenceClient;
import java.time.Duration;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

/** Concurrent "open this flagged transaction" operation. When an analyst selects a node we want both
 *  the faithful explanation AND (ideally) its grounded LLM summary in one round-trip. Those are two
 *  independent interactions with the model service, so we fan them out on virtual threads and join
 *  with deadlines:
 *
 *    - the capped explanation (also kicks off summary generation on the inference side), and
 *    - a short bounded wait for the hosted-LLM summary.
 *
 *  If the summary lands inside the wait, it's merged in and the UI shows it immediately (no polling);
 *  otherwise the response keeps {@code summary_pending=true} and the UI falls back to polling
 *  /api/summary. This is the score+explain+summary fan-out done structurally rather than with manual
 *  CompletableFuture juggling. (StructuredTaskScope / JEP 453 is the eventual idiom; it's still a
 *  preview API, so we stay on GA virtual threads via a virtual-thread-per-task executor.) */
@Service
public class InvestigationService {

    private static final Logger log = LoggerFactory.getLogger(InvestigationService.class);
    private static final Duration SUMMARY_WAIT = Duration.ofSeconds(3);   // brief race for the LLM
    private static final long EXPLAIN_DEADLINE_S = 40;

    private final AnalysisService analysis;
    private final InferenceClient inference;

    public InvestigationService(AnalysisService analysis, InferenceClient inference) {
        this.analysis = analysis;
        this.inference = inference;
    }

    /** Explanation + (best-effort, time-boxed) AI summary, fetched concurrently. */
    public JsonNode investigate(Long datasetId, int nodeId) {
        try (var scope = Executors.newVirtualThreadPerTaskExecutor()) {
            Future<JsonNode> explanationF = scope.submit(() -> analysis.explain(datasetId, nodeId));
            Future<String> summaryF = scope.submit(() -> awaitSummary(nodeId, SUMMARY_WAIT));

            JsonNode explanation = explanationF.get(EXPLAIN_DEADLINE_S, TimeUnit.SECONDS);
            if (!(explanation instanceof ObjectNode obj)) {
                return explanation;   // defensive: contracts are always objects
            }
            String summary = joinSummary(summaryF);
            if (summary != null && !summary.isBlank()) {
                obj.put("summary", summary);
                obj.put("summary_pending", false);   // ready now -> UI shows it, no polling needed
            }
            // else: leave the instant template summary + summary_pending=true; the UI polls /api/summary
            return obj;
        } catch (TimeoutException e) {
            throw new InferenceUnavailableException("explanation timed out for node " + nodeId, e);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause();
            if (cause instanceof RuntimeException re) throw re;   // preserve 404 / 502 mapping
            throw new InferenceUnavailableException("investigate failed for node " + nodeId, cause);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new InferenceUnavailableException("investigate interrupted for node " + nodeId, e);
        }
    }

    private String joinSummary(Future<String> summaryF) {
        try {
            return summaryF.get(SUMMARY_WAIT.toSeconds() + 1, TimeUnit.SECONDS);
        } catch (TimeoutException | ExecutionException e) {
            summaryF.cancel(true);
            return null;   // not ready in time (or errored) -> UI will poll
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return null;
        }
    }

    /** Poll the inference summary endpoint until ready or the budget elapses. */
    private String awaitSummary(int nodeId, Duration budget) throws InterruptedException {
        long deadline = System.nanoTime() + budget.toNanos();
        while (System.nanoTime() < deadline) {
            try {
                JsonNode s = inference.summary(nodeId);
                if (s != null && s.path("ready").asBoolean(false) && s.hasNonNull("summary")) {
                    return s.get("summary").asText();
                }
            } catch (RuntimeException e) {
                log.debug("summary poll failed for node {}: {}", nodeId, e.toString());
                return null;   // don't hammer a failing dependency; UI can still poll later
            }
            Thread.sleep(400);
        }
        return null;
    }
}
