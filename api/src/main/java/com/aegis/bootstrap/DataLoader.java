package com.aegis.bootstrap;

import com.aegis.client.InferenceClient;
import com.aegis.entity.Dataset;
import com.aegis.exception.InferenceUnavailableException;
import com.aegis.repository.DatasetRepository;
import com.fasterxml.jackson.databind.JsonNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.ApplicationRunner;
import org.springframework.boot.ApplicationArguments;
import org.springframework.stereotype.Component;

/** Seeds the dataset registry from the inference service's /health on startup, so the demo has a
 *  dataset to pick without manual DB setup (spec §8.5 / §12 Phase 8 instant demo). */
@Component
public class DataLoader implements ApplicationRunner {

    private static final Logger log = LoggerFactory.getLogger(DataLoader.class);
    private final DatasetRepository repo;
    private final InferenceClient inference;

    public DataLoader(DatasetRepository repo, InferenceClient inference) {
        this.repo = repo; this.inference = inference;
    }

    @Override
    public void run(ApplicationArguments args) {
        try {
            JsonNode info = inference.info();
            String name = info.path("dataset").asText("LI-Small");
            if (repo.findByName(name).isEmpty()) {
                repo.save(new Dataset(name, info.path("model").asText("gnn"),
                        info.path("num_nodes").asLong(), info.path("num_edges").asLong(),
                        info.path("num_illicit").asLong()));
                log.info("Seeded dataset '{}' from the inference service", name);
            }
        } catch (InferenceUnavailableException e) {
            log.warn("Inference service not reachable at startup; dataset registry left empty. {}",
                    e.getMessage());
        }
    }
}
