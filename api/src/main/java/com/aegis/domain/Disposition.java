package com.aegis.domain;

/** Terminal outcome recorded when a case is closed. */
public enum Disposition {
    SAR_FILED(CaseEvent.CLOSE_SAR, CaseState.CLOSED_SAR),
    FALSE_POSITIVE(CaseEvent.CLOSE_FALSE_POSITIVE, CaseState.CLOSED_FALSE_POSITIVE),
    NO_ACTION(CaseEvent.CLOSE_NO_ACTION, CaseState.CLOSED_NO_ACTION);

    private final CaseEvent event;
    private final CaseState terminalState;

    Disposition(CaseEvent event, CaseState terminalState) {
        this.event = event; this.terminalState = terminalState;
    }

    public CaseEvent event() { return event; }
    public CaseState terminalState() { return terminalState; }
}
