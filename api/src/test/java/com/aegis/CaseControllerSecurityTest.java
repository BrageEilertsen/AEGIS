package com.aegis.controller;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.aegis.config.SecurityConfig;
import com.aegis.domain.Priority;
import com.aegis.entity.CaseFile;
import com.aegis.service.CaseService;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.context.annotation.Import;
import org.springframework.security.test.context.support.WithAnonymousUser;
import org.springframework.security.test.context.support.WithMockUser;
import org.springframework.test.web.servlet.MockMvc;

/** RBAC on the case API: must be a known analyst to act; only reviewers/admins may close a case
 *  (segregation of duties). */
@WebMvcTest(CaseController.class)
@Import(SecurityConfig.class)
class CaseControllerSecurityTest {

    @Autowired MockMvc mvc;
    @MockBean CaseService service;

    private CaseFile sampleCase() {
        when(service.list(any(), any(), org.mockito.ArgumentMatchers.anyInt())).thenReturn(List.of());
        when(service.transition(anyLong(), any(), anyString(), any()))
                .thenReturn(new CaseFile("c", Priority.HIGH));
        return new CaseFile("c", Priority.HIGH);
    }

    @Test
    @WithAnonymousUser
    void anonymousIsForbidden() throws Exception {
        mvc.perform(get("/api/cases")).andExpect(status().isForbidden());
    }

    @Test
    @WithMockUser(roles = "ANALYST")
    void analystCanListCases() throws Exception {
        sampleCase();
        mvc.perform(get("/api/cases")).andExpect(status().isOk());
    }

    @Test
    @WithMockUser(roles = "ANALYST")
    void analystCannotClose() throws Exception {
        sampleCase();
        mvc.perform(post("/api/cases/1/disposition").contentType("application/json")
                        .content("{\"disposition\":\"SAR_FILED\",\"rationale\":\"x\"}"))
                .andExpect(status().isForbidden());
    }

    @Test
    @WithMockUser(roles = "REVIEWER")
    void reviewerCanClose() throws Exception {
        sampleCase();
        mvc.perform(post("/api/cases/1/disposition").contentType("application/json")
                        .content("{\"disposition\":\"SAR_FILED\",\"rationale\":\"x\"}"))
                .andExpect(status().isOk());
    }
}
