package com.aegis.domain;

import java.time.Duration;

/** Case priority, which also drives the investigation SLA window. */
public enum Priority {
    LOW(Duration.ofDays(5)),
    MEDIUM(Duration.ofDays(3)),
    HIGH(Duration.ofHours(24)),
    CRITICAL(Duration.ofHours(4));

    private final Duration sla;

    Priority(Duration sla) { this.sla = sla; }

    public Duration sla() { return sla; }
}
