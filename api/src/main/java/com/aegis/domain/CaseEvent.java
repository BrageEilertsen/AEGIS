package com.aegis.domain;

/** Events that drive case state transitions (the edges of the workflow state machine). */
public enum CaseEvent {
    ASSIGN,                 // NEW -> ASSIGNED
    START_INVESTIGATION,    // ASSIGNED -> IN_INVESTIGATION
    SUBMIT_FOR_REVIEW,      // IN_INVESTIGATION -> PENDING_REVIEW
    RETURN_FOR_REWORK,      // PENDING_REVIEW -> IN_INVESTIGATION
    CLOSE_SAR,              // PENDING_REVIEW -> CLOSED_SAR
    CLOSE_FALSE_POSITIVE,   // PENDING_REVIEW -> CLOSED_FALSE_POSITIVE
    CLOSE_NO_ACTION         // PENDING_REVIEW -> CLOSED_NO_ACTION
}
