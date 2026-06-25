package com.aegis.controller;

import com.aegis.dto.FlagDto;
import com.aegis.dto.MetricsDto;
import com.aegis.service.AnalysisService;
import com.fasterxml.jackson.databind.JsonNode;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.Pattern;
import java.util.List;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

/** Flagged transactions, explanations (capped subgraph), metrics, and the adversarial before/after.
 *  datasetId is part of the path for fidelity to the spec; the demo serves a single loaded model. */
@RestController
@RequestMapping("/api")
@Validated
public class AnalysisController {

    private final AnalysisService service;
    private final com.aegis.service.InvestigationService investigation;

    public AnalysisController(AnalysisService service,
                             com.aegis.service.InvestigationService investigation) {
        this.service = service;
        this.investigation = investigation;
    }

    @GetMapping("/flags/{datasetId}")
    public List<FlagDto> flags(@PathVariable Long datasetId,
                               @RequestParam(defaultValue = "0.5") @Min(0) @Max(1) double threshold,
                               @RequestParam(defaultValue = "100") @Min(1) @Max(5000) int limit) {
        return service.flags(threshold, limit);
    }

    /** The renderable, capped neighbourhood + faithful explanation for one flagged transaction. */
    @GetMapping("/explain/{datasetId}/{nodeId}")
    public JsonNode explain(@PathVariable @Min(1) Long datasetId,
                            @PathVariable @Min(0) @Max(100_000_000) int nodeId) {
        return service.explain(datasetId, nodeId);
    }

    /** Open a flagged transaction: the explanation plus its AI summary if it's ready within a short
     *  window — both fetched concurrently on virtual threads, so the analyst usually gets the whole
     *  picture in one round-trip (falling back to /api/summary polling when the LLM is slower). */
    @GetMapping("/investigate/{datasetId}/{nodeId}")
    public JsonNode investigate(@PathVariable @Min(1) Long datasetId,
                                @PathVariable @Min(0) @Max(100_000_000) int nodeId) {
        return investigation.investigate(datasetId, nodeId);
    }

    /** Alias: the capped flagged subgraph for a node (spec §8.4) is the explanation's neighbourhood. */
    @GetMapping("/graph/{datasetId}/{nodeId}")
    public JsonNode graph(@PathVariable @Min(1) Long datasetId,
                          @PathVariable @Min(0) @Max(100_000_000) int nodeId) {
        return service.explain(datasetId, nodeId).get("neighborhood_subgraph");
    }

    /** Async LLM narration for a node ({ready, summary}); the UI polls this to upgrade the
     *  instant template summary shown with the explanation. */
    @GetMapping("/summary/{datasetId}/{nodeId}")
    public JsonNode summary(@PathVariable @Min(1) Long datasetId,
                            @PathVariable @Min(0) @Max(100_000_000) int nodeId) {
        return service.summary(nodeId);
    }

    @GetMapping("/metrics/{datasetId}")
    public MetricsDto metrics(@PathVariable @Min(1) Long datasetId,
                              @RequestParam(defaultValue = "test") @Pattern(regexp = "val|test") String split) {
        return service.metrics(split);
    }

    @PostMapping("/adversarial/run")
    public JsonNode adversarial() {
        return service.adversarial();
    }
}
