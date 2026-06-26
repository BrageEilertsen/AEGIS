package com.aegis.security;

import static org.mockito.Mockito.when;
import static org.springframework.security.test.web.servlet.request.SecurityMockMvcRequestPostProcessors.jwt;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.aegis.config.SecurityConfig;
import com.aegis.controller.AnalysisController;
import com.aegis.controller.MeController;
import com.aegis.dto.FlagDto;
import com.aegis.service.AnalysisService;
import com.aegis.service.InvestigationService;
import java.util.List;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.context.annotation.Import;
import org.springframework.security.oauth2.jwt.JwtDecoder;
import org.springframework.test.web.servlet.MockMvc;

/** Verifies the failsafe security model: the public read tier is always open, and the analyst probe
 *  (/api/me) is gated only when auth is enabled (a JwtDecoder is present). */
class SecurityConfigTest {

    @WebMvcTest(controllers = {MeController.class, AnalysisController.class})
    @Import(SecurityConfig.class)
    @Nested
    class AuthDisabled {   // no JwtDecoder bean -> security wide open (the public demo state)
        @Autowired MockMvc mvc;
        @MockBean AnalysisService analysis;
        @MockBean InvestigationService investigation;

        @Test
        void publicEndpointOpen() throws Exception {
            when(analysis.flags(0.5, 100)).thenReturn(List.of(new FlagDto(7, 0.9, 1)));
            mvc.perform(get("/api/flags/1?threshold=0.5&limit=100")).andExpect(status().isOk());
        }

        @Test
        void meReportsAnonymous() throws Exception {
            mvc.perform(get("/api/me"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.authenticated").value(false));
        }
    }

    @WebMvcTest(controllers = {MeController.class, AnalysisController.class})
    @Import(SecurityConfig.class)
    @Nested
    class AuthEnabled {   // a (mock) JwtDecoder present -> resource server active
        @Autowired MockMvc mvc;
        @MockBean AnalysisService analysis;
        @MockBean InvestigationService investigation;
        @MockBean JwtDecoder jwtDecoder;

        @Test
        void publicEndpointStillOpen() throws Exception {
            when(analysis.flags(0.5, 100)).thenReturn(List.of(new FlagDto(7, 0.9, 1)));
            mvc.perform(get("/api/flags/1?threshold=0.5&limit=100")).andExpect(status().isOk());
        }

        @Test
        void meRequiresAuth() throws Exception {
            mvc.perform(get("/api/me")).andExpect(status().isUnauthorized());
        }

        @Test
        void meAllowsValidToken() throws Exception {
            mvc.perform(get("/api/me").with(jwt().jwt(j -> j.subject("analyst@bank"))))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.authenticated").value(true));
        }
    }
}
