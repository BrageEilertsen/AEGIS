package com.aegis;

import static com.github.tomakehurst.wiremock.client.WireMock.aResponse;
import static com.github.tomakehurst.wiremock.client.WireMock.get;
import static com.github.tomakehurst.wiremock.client.WireMock.okJson;
import static com.github.tomakehurst.wiremock.client.WireMock.urlPathEqualTo;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.aegis.bootstrap.DataLoader;
import com.aegis.client.InferenceClient;
import com.aegis.exception.InferenceUnavailableException;
import com.github.tomakehurst.wiremock.WireMockServer;
import com.github.tomakehurst.wiremock.core.WireMockConfiguration;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;

/** Verifies the BFF↔inference contract and resilience by stubbing the FastAPI service with WireMock
 *  (in-JVM, no Docker): a healthy stub returns flags; a failing stub trips the client's
 *  retry/circuit-breaker and surfaces a clean InferenceUnavailableException. */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.NONE, properties = {
        "spring.datasource.url=jdbc:h2:mem:wmit;MODE=PostgreSQL;DB_CLOSE_DELAY=-1",
        "spring.datasource.driver-class-name=org.h2.Driver",
        "spring.jpa.hibernate.ddl-auto=create-drop",
        "AEGIS_OIDC_ISSUER="
})
class WireMockInferenceTest {

    static final WireMockServer wm = new WireMockServer(WireMockConfiguration.options().dynamicPort());

    @MockBean DataLoader dataLoader;
    @Autowired InferenceClient inference;

    @DynamicPropertySource
    static void inferenceUrl(DynamicPropertyRegistry registry) {
        wm.start();
        registry.add("aegis.inference.base-url", () -> "http://localhost:" + wm.port());
    }

    @AfterAll
    static void stop() { wm.stop(); }

    @Test
    void returnsFlagsFromInference() {
        wm.stubFor(get(urlPathEqualTo("/flags"))
                .willReturn(okJson("[{\"node_id\":7,\"score\":0.97,\"label\":1}]")));

        assertThat(inference.flags(0.5, 100)).hasSize(1)
                .first().satisfies(f -> assertThat(f.nodeId()).isEqualTo(7));
    }

    @Test
    void surfacesCleanErrorWhenInferenceFails() {
        wm.stubFor(get(urlPathEqualTo("/metrics"))
                .willReturn(aResponse().withStatus(500)));

        assertThatThrownBy(() -> inference.metrics("test"))
                .isInstanceOf(InferenceUnavailableException.class);
    }
}
