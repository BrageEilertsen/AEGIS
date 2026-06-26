package com.aegis.workflow;

import static com.aegis.domain.CaseEvent.*;
import static com.aegis.domain.CaseState.*;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.aegis.exception.IllegalCaseTransitionException;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

/** The state machine accepts the legal lifecycle path and rejects out-of-order events. */
@SpringBootTest(classes = {CaseStateMachineConfig.class, CaseWorkflow.class})
class CaseWorkflowTest {

    @Autowired CaseWorkflow workflow;

    @Test
    void walksTheLegalPath() {
        assertThat(workflow.apply(NEW, ASSIGN)).isEqualTo(ASSIGNED);
        assertThat(workflow.apply(ASSIGNED, START_INVESTIGATION)).isEqualTo(IN_INVESTIGATION);
        assertThat(workflow.apply(IN_INVESTIGATION, SUBMIT_FOR_REVIEW)).isEqualTo(PENDING_REVIEW);
        assertThat(workflow.apply(PENDING_REVIEW, RETURN_FOR_REWORK)).isEqualTo(IN_INVESTIGATION);
        assertThat(workflow.apply(PENDING_REVIEW, CLOSE_SAR)).isEqualTo(CLOSED_SAR);
        assertThat(workflow.apply(PENDING_REVIEW, CLOSE_FALSE_POSITIVE)).isEqualTo(CLOSED_FALSE_POSITIVE);
    }

    @Test
    void rejectsOutOfOrderEvents() {
        assertThatThrownBy(() -> workflow.apply(NEW, CLOSE_SAR))
                .isInstanceOf(IllegalCaseTransitionException.class);
        assertThatThrownBy(() -> workflow.apply(ASSIGNED, CLOSE_NO_ACTION))
                .isInstanceOf(IllegalCaseTransitionException.class);
        assertThatThrownBy(() -> workflow.apply(IN_INVESTIGATION, ASSIGN))
                .isInstanceOf(IllegalCaseTransitionException.class);
    }
}
