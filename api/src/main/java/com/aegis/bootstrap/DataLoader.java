package com.aegis.bootstrap;

import com.aegis.client.InferenceClient;
import com.aegis.entity.Dataset;
import com.aegis.repository.DatasetRepository;
import com.fasterxml.jackson.databind.JsonNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.ApplicationRunner;
import org.springframework.boot.ApplicationArguments;
import org.springframework.stereotype.Component;

/** Seeds the dataset registry from the inference service's /info on startup, so the demo has a
 *  dataset to pick without manual DB setup (spec §8.5 / §12 Phase 8 instant demo).
 *
 *  Resilient by design: it retries while the (large) inference image warms up, and NEVER lets an
 *  exception escape — a failed seed must not crash the API (it just leaves the registry to be
 *  seeded on a later start). */
@Component
public class DataLoader implements ApplicationRunner {

    private static final Logger log = LoggerFactory.getLogger(DataLoader.class);
    private static final int MAX_ATTEMPTS = 20;
    private static final long RETRY_MS = 6000;

    private final DatasetRepository repo;
    private final InferenceClient inference;

    public DataLoader(DatasetRepository repo, InferenceClient inference) {
        this.repo = repo; this.inference = inference;
    }

    @Override
    public void run(ApplicationArguments args) {
        for (int attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
            try {
                JsonNode info = inference.info();
                if (info != null && !info.isMissingNode() && info.has("dataset")) {
                    String name = info.path("dataset").asText("LI-Small");
                    if (repo.findByName(name).isEmpty()) {
                        repo.save(new Dataset(name, info.path("model").asText("gnn"),
                                info.path("num_nodes").asLong(), info.path("num_edges").asLong(),
                                info.path("num_illicit").asLong()));
                        log.info("Seeded dataset '{}' from the inference service", name);
                    }
                    return;
                }
                log.warn("Inference /info not ready yet (attempt {}/{}); retrying…", attempt, MAX_ATTEMPTS);
            } catch (Exception e) {
                log.warn("Inference not reachable at startup (attempt {}/{}): {}", attempt, MAX_ATTEMPTS, e.getMessage());
            }
            try {
                Thread.sleep(RETRY_MS);
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                return;
            }
        }
        log.warn("Could not seed the dataset from inference after {} attempts; it will seed on a later start.", MAX_ATTEMPTS);
    }
}
