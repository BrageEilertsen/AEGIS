package com.aegis.controller;

import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

import com.aegis.dto.FlagDto;
import com.aegis.dto.MetricsDto;
import com.aegis.service.AnalysisService;
import com.aegis.service.InvestigationService;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.web.servlet.MockMvc;

@WebMvcTest(AnalysisController.class)
@AutoConfigureMockMvc(addFilters = false)   // pure controller slice — skip the rate-limit filter
class AnalysisControllerTest {

    @Autowired private MockMvc mvc;
    @MockBean private AnalysisService service;
    @MockBean private InvestigationService investigation;

    @Test
    void flagsReturnsRankedTransactions() throws Exception {
        when(service.flags(0.5, 100)).thenReturn(List.of(new FlagDto(7, 0.97, 1)));
        mvc.perform(get("/api/flags/1?threshold=0.5&limit=100"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].node_id").value(7))
                .andExpect(jsonPath("$[0].score").value(0.97));
    }

    @Test
    void metricsExposePrAucNotAccuracy() throws Exception {
        when(service.metrics("test")).thenReturn(new MetricsDto(0.24, 0.88, 0.0, 0.9, 0.0,
                36363, 915, Map.of("tn", 35448, "fp", 0, "fn", 915, "tp", 0)));
        mvc.perform(get("/api/metrics/1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.pr_auc").value(0.24))
                .andExpect(jsonPath("$.roc_auc").value(0.88));
    }

    @Test
    void rejectsOutOfRangeThreshold() throws Exception {
        mvc.perform(get("/api/flags/1?threshold=2.0")).andExpect(status().isBadRequest());
    }
}
