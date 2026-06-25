package com.aegis.domain;

/** Lifecycle states of an AML case (the nodes of the workflow state machine). */
public enum CaseState {
    NEW,                       // created from one or more alerts, not yet picked up
    ASSIGNED,                  // an analyst owns it
    IN_INVESTIGATION,          // active investigation
    PENDING_REVIEW,            // analyst submitted a recommendation; awaiting a reviewer
    CLOSED_SAR,                // closed — Suspicious Activity Report filed
    CLOSED_FALSE_POSITIVE,     // closed — not suspicious
    CLOSED_NO_ACTION;          // closed — suspicious but no action warranted

    public boolean isClosed() {
        return this == CLOSED_SAR || this == CLOSED_FALSE_POSITIVE || this == CLOSED_NO_ACTION;
    }
}
