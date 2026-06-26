package com.aegis;

import static org.assertj.core.api.Assertions.assertThat;

import com.aegis.bootstrap.DataLoader;
import com.aegis.domain.CaseEvent;
import com.aegis.domain.CaseState;
import com.aegis.domain.Priority;
import com.aegis.entity.CaseFile;
import com.aegis.service.AuditService;
import com.aegis.service.CaseService;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

/** Integration test against a REAL PostgreSQL (Testcontainers), not H2 — proves the JPA schema,
 *  the JSONB explanation column, the case tables and the audit hash chain all work on the actual
 *  database the app deploys to. Runs in CI (Docker); a *IT name so failsafe owns it, not surefire. */
@Testcontainers
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.NONE, properties = "AEGIS_OIDC_ISSUER=")
class PostgresPersistenceIT {

    @Container
    static final PostgreSQLContainer<?> postgres = new PostgreSQLContainer<>("postgres:16-alpine");

    @DynamicPropertySource
    static void datasource(DynamicPropertyRegistry registry) {
        registry.add("spring.datasource.url", postgres::getJdbcUrl);
        registry.add("spring.datasource.username", postgres::getUsername);
        registry.add("spring.datasource.password", postgres::getPassword);
        registry.add("spring.jpa.hibernate.ddl-auto", () -> "create-drop");
    }

    @MockBean DataLoader dataLoader;
    @Autowired CaseService cases;
    @Autowired AuditService audit;

    @Test
    void caseLifecyclePersistsOnRealPostgres() {
        CaseFile c = cases.createCase("PG integration case", Priority.HIGH, List.of(), "analyst1");
        cases.assign(c.getId(), "analyst1", "lead");
        cases.transition(c.getId(), CaseEvent.START_INVESTIGATION, "analyst1", null);
        cases.transition(c.getId(), CaseEvent.SUBMIT_FOR_REVIEW, "analyst1", "ready");
        c = cases.transition(c.getId(), CaseEvent.CLOSE_SAR, "reviewer1", "filing");

        assertThat(c.getState()).isEqualTo(CaseState.CLOSED_SAR);
        // every action was hash-chained, and the chain verifies clean on real Postgres
        assertThat(audit.verify().valid()).isTrue();
        assertThat(audit.forSubject("CASE", c.getId())).isNotEmpty();
    }
}
