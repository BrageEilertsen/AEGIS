package com.aegis.service;

import static com.aegis.domain.CaseEvent.*;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.aegis.bootstrap.DataLoader;
import com.aegis.domain.CaseState;
import com.aegis.domain.Disposition;
import com.aegis.domain.Priority;
import com.aegis.entity.Alert;
import com.aegis.entity.CaseFile;
import com.aegis.exception.IllegalCaseTransitionException;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;

/** End-to-end case lifecycle against a real (H2) DB: alert -> case -> assign -> investigate ->
 *  review -> SAR, with SLA set on start, disposition recorded on close, and history captured. */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.NONE, properties = {
        "spring.datasource.url=jdbc:h2:mem:caseit;MODE=PostgreSQL;DB_CLOSE_DELAY=-1",
        "spring.datasource.driver-class-name=org.h2.Driver",
        "spring.jpa.hibernate.ddl-auto=create-drop",
        "AEGIS_OIDC_ISSUER="
})
class CaseServiceTest {

    @MockBean DataLoader dataLoader;
    @Autowired CaseService service;

    @Test
    void fullLifecycleToSar() {
        Alert a = service.createAlert(1L, 42, 0.99, "circular");
        CaseFile c = service.createCase("Suspected layering #42", Priority.HIGH, List.of(a.getId()), "analyst1");
        assertThat(c.getState()).isEqualTo(CaseState.NEW);
        assertThat(service.alertsFor(c.getId())).extracting(Alert::getId).containsExactly(a.getId());

        service.assign(c.getId(), "analyst1", "lead");
        c = service.transition(c.getId(), START_INVESTIGATION, "analyst1", null);
        assertThat(c.getState()).isEqualTo(CaseState.IN_INVESTIGATION);
        assertThat(c.getSlaDueAt()).isNotNull();              // SLA window opened

        service.transition(c.getId(), SUBMIT_FOR_REVIEW, "analyst1", "looks like layering");
        c = service.transition(c.getId(), CLOSE_SAR, "reviewer1", "filing SAR");

        assertThat(c.getState()).isEqualTo(CaseState.CLOSED_SAR);
        assertThat(c.getDisposition()).isEqualTo(Disposition.SAR_FILED);
        assertThat(c.getClosedBy()).isEqualTo("reviewer1");
        assertThat(service.activitiesFor(c.getId()))
                .extracting(x -> x.getAction())
                .contains("CREATED", "ASSIGN", "START_INVESTIGATION", "SUBMIT_FOR_REVIEW", "CLOSE_SAR");
    }

    @Test
    void illegalTransitionRejected() {
        CaseFile c = service.createCase("bad path", Priority.LOW, List.of(), "analyst1");
        assertThatThrownBy(() -> service.transition(c.getId(), CLOSE_SAR, "x", null))
                .isInstanceOf(IllegalCaseTransitionException.class);
    }
}
