package com.aegis.workflow;

import com.aegis.domain.CaseEvent;
import com.aegis.domain.CaseState;
import com.aegis.exception.IllegalCaseTransitionException;
import java.util.List;
import org.springframework.messaging.support.MessageBuilder;
import org.springframework.statemachine.StateMachine;
import org.springframework.statemachine.StateMachineEventResult;
import org.springframework.statemachine.config.StateMachineFactory;
import org.springframework.statemachine.support.DefaultStateMachineContext;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

/** Drives a single case's transition through the shared Spring State Machine definition. A fresh
 *  machine is taken from the factory, reset to the case's persisted state, and the event applied;
 *  the resulting state is returned, or an {@link IllegalCaseTransitionException} thrown if the event
 *  is not legal from the current state. (The state lives in the entity — the machine is stateless
 *  infrastructure, rehydrated per call.) */
@Component
public class CaseWorkflow {

    private final StateMachineFactory<CaseState, CaseEvent> factory;

    public CaseWorkflow(StateMachineFactory<CaseState, CaseEvent> factory) {
        this.factory = factory;
    }

    public CaseState apply(CaseState current, CaseEvent event) {
        StateMachine<CaseState, CaseEvent> sm = factory.getStateMachine();
        sm.stopReactively().block();
        sm.getStateMachineAccessor().doWithAllRegions(access ->
                access.resetStateMachineReactively(
                        new DefaultStateMachineContext<>(current, null, null, null)).block());
        sm.startReactively().block();

        List<StateMachineEventResult<CaseState, CaseEvent>> results = sm
                .sendEvent(Mono.just(MessageBuilder.withPayload(event).build()))
                .collectList().block();

        boolean accepted = results != null && results.stream()
                .anyMatch(r -> r.getResultType() == StateMachineEventResult.ResultType.ACCEPTED);
        if (!accepted) {
            throw new IllegalCaseTransitionException(
                    "Cannot %s a case in state %s".formatted(event, current));
        }
        return sm.getState().getId();
    }
}
