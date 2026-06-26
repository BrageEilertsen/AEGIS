package com.aegis;

import static org.assertj.core.api.Assertions.assertThat;

import com.aegis.bootstrap.DataLoader;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;

/** Full-context test over a real servlet container with virtual threads on (mirrors the live
 *  deployment), which a @WebMvcTest slice can't reproduce. Guards the two bugs that only showed up
 *  in production: /api/me must report anonymous (not 500) and an unmapped route must be 404. */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT, properties = {
        "spring.threads.virtual.enabled=true",
        "spring.datasource.url=jdbc:h2:mem:meit;MODE=PostgreSQL;DB_CLOSE_DELAY=-1",
        "spring.datasource.driver-class-name=org.h2.Driver",
        "spring.jpa.hibernate.ddl-auto=create-drop",
        "AEGIS_OIDC_ISSUER="   // public/demo mode (auth disabled)
})
class MeEndpointTest {

    @MockBean DataLoader dataLoader;   // skip the startup inference-seeding runner

    @Autowired TestRestTemplate rest;

    @Test
    void mePublicReturnsAnonymous() {
        ResponseEntity<String> r = rest.getForEntity("/api/me", String.class);
        assertThat(r.getStatusCode().value()).isEqualTo(200);
        assertThat(r.getBody()).contains("\"authenticated\":false");
    }

    @Test
    void unmappedRouteIsNotFound() {
        // A genuinely unmapped path -> 404 (the catch-all must not turn NoResourceFound into 500).
        ResponseEntity<String> r = rest.getForEntity("/api/does-not-exist-xyz", String.class);
        assertThat(r.getStatusCode().value()).isEqualTo(404);
    }
}
