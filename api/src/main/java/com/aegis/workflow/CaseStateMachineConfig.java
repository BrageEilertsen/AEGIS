package com.aegis.workflow;

import com.aegis.domain.CaseEvent;
import com.aegis.domain.CaseState;
import org.springframework.context.annotation.Configuration;
import org.springframework.statemachine.config.EnableStateMachineFactory;
import org.springframework.statemachine.config.StateMachineConfigurerAdapter;
import org.springframework.statemachine.config.builders.StateMachineStateConfigurer;
import org.springframework.statemachine.config.builders.StateMachineTransitionConfigurer;

/** The AML case lifecycle as a Spring State Machine: which transitions are legal, independent of who
 *  may perform them (authorization is enforced separately by method security). A case can only ever
 *  move along these edges, so an out-of-order action (e.g. closing a case that was never reviewed) is
 *  rejected structurally rather than by ad-hoc if-checks. */
@Configuration
@EnableStateMachineFactory
public class CaseStateMachineConfig extends StateMachineConfigurerAdapter<CaseState, CaseEvent> {

    @Override
    public void configure(StateMachineStateConfigurer<CaseState, CaseEvent> states) throws Exception {
        states.withStates()
                .initial(CaseState.NEW)
                .state(CaseState.ASSIGNED)
                .state(CaseState.IN_INVESTIGATION)
                .state(CaseState.PENDING_REVIEW)
                .end(CaseState.CLOSED_SAR)
                .end(CaseState.CLOSED_FALSE_POSITIVE)
                .end(CaseState.CLOSED_NO_ACTION);
    }

    @Override
    public void configure(StateMachineTransitionConfigurer<CaseState, CaseEvent> t) throws Exception {
        t.withExternal().source(CaseState.NEW).target(CaseState.ASSIGNED).event(CaseEvent.ASSIGN)
         .and().withExternal().source(CaseState.ASSIGNED).target(CaseState.IN_INVESTIGATION).event(CaseEvent.START_INVESTIGATION)
         .and().withExternal().source(CaseState.IN_INVESTIGATION).target(CaseState.PENDING_REVIEW).event(CaseEvent.SUBMIT_FOR_REVIEW)
         .and().withExternal().source(CaseState.PENDING_REVIEW).target(CaseState.IN_INVESTIGATION).event(CaseEvent.RETURN_FOR_REWORK)
         .and().withExternal().source(CaseState.PENDING_REVIEW).target(CaseState.CLOSED_SAR).event(CaseEvent.CLOSE_SAR)
         .and().withExternal().source(CaseState.PENDING_REVIEW).target(CaseState.CLOSED_FALSE_POSITIVE).event(CaseEvent.CLOSE_FALSE_POSITIVE)
         .and().withExternal().source(CaseState.PENDING_REVIEW).target(CaseState.CLOSED_NO_ACTION).event(CaseEvent.CLOSE_NO_ACTION);
    }
}
