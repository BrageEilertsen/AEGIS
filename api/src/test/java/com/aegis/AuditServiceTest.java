package com.aegis.service;

import static org.assertj.core.api.Assertions.assertThat;

import com.aegis.bootstrap.DataLoader;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.jdbc.core.JdbcTemplate;

/** The audit chain verifies clean when intact, and verification fails (pointing to the row) when a
 *  past record is tampered with directly in the database — the whole point of a hash chain. */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.NONE, properties = {
        "spring.datasource.url=jdbc:h2:mem:auditit;MODE=PostgreSQL;DB_CLOSE_DELAY=-1",
        "spring.datasource.driver-class-name=org.h2.Driver",
        "spring.jpa.hibernate.ddl-auto=create-drop",
        "AEGIS_OIDC_ISSUER="
})
class AuditServiceTest {

    @MockBean DataLoader dataLoader;
    @Autowired AuditService audit;
    @Autowired JdbcTemplate jdbc;

    @org.junit.jupiter.api.BeforeEach
    void clean() {   // the methods share one in-memory DB — start each from an empty chain
        jdbc.update("DELETE FROM audit_record");
        jdbc.update("DELETE FROM audit_head");
    }

    @Test
    void intactChainVerifies() {
        audit.append("analyst1", "CASE_CREATED", "CASE", 1L, "{\"a\":1}");
        audit.append("analyst1", "CASE_ASSIGNED", "CASE", 1L, "{\"to\":\"analyst1\"}");
        audit.append("reviewer1", "CASE_CLOSE_SAR", "CASE", 1L, "{\"disposition\":\"SAR_FILED\"}");

        AuditService.Verification v = audit.verify();
        assertThat(v.valid()).isTrue();
        assertThat(v.records()).isEqualTo(3);
        assertThat(audit.forSubject("CASE", 1L)).hasSize(3);
    }

    @Test
    void tamperingIsDetected() {
        audit.append("analyst1", "CASE_CREATED", "CASE", 7L, "{\"a\":1}");
        audit.append("analyst1", "CASE_ASSIGNED", "CASE", 7L, "{\"to\":\"analyst1\"}");
        audit.append("reviewer1", "CASE_CLOSE_SAR", "CASE", 7L, "{\"disposition\":\"NO_ACTION\"}");

        // Forge a past decision directly in the DB (bypassing append/hashing).
        jdbc.update("UPDATE audit_record SET payload = ? WHERE seq = ?", "{\"disposition\":\"SAR_FILED\"}", 3);

        AuditService.Verification v = audit.verify();
        assertThat(v.valid()).isFalse();
        assertThat(v.brokenAtSeq()).isEqualTo(3);
        assertThat(v.reason()).contains("altered");
    }
}
