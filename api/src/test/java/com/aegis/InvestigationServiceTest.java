package com.aegis.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.when;

import com.aegis.client.InferenceClient;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

/** The investigate() fan-out merges the AI summary into the explanation when it's ready in time, and
 *  otherwise leaves the instant template + summary_pending=true for the UI to poll. */
@ExtendWith(MockitoExtension.class)
class InvestigationServiceTest {

    private final ObjectMapper mapper = new ObjectMapper();

    @Mock AnalysisService analysis;
    @Mock InferenceClient inference;
    @InjectMocks InvestigationService service;

    private ObjectNode explanation() {
        return mapper.createObjectNode()
                .put("node_id", 7).put("score", 0.99)
                .put("summary", "template summary").put("summary_pending", true);
    }

    @Test
    void mergesSummaryWhenReadyInTime() {
        when(analysis.explain(1L, 7)).thenReturn(explanation());
        ObjectNode ready = mapper.createObjectNode().put("ready", true).put("summary", "LLM summary");
        when(inference.summary(7)).thenReturn((JsonNode) ready);

        JsonNode out = service.investigate(1L, 7);

        assertThat(out.get("summary").asText()).isEqualTo("LLM summary");
        assertThat(out.get("summary_pending").asBoolean()).isFalse();
    }

    @Test
    void keepsTemplateAndPendingWhenSummaryNotReady() {
        when(analysis.explain(1L, 7)).thenReturn(explanation());
        ObjectNode notReady = mapper.createObjectNode().put("ready", false).putNull("summary");
        when(inference.summary(7)).thenReturn((JsonNode) notReady);

        JsonNode out = service.investigate(1L, 7);

        assertThat(out.get("summary").asText()).isEqualTo("template summary");
        assertThat(out.get("summary_pending").asBoolean()).isTrue();
    }
}
