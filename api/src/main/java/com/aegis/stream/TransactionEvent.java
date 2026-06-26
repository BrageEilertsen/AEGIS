package com.aegis.stream;

import java.time.Instant;

/** One transaction arriving on the stream (from the simulated feed or a real Kafka topic). */
public record TransactionEvent(String id, Instant ts, String source, String target,
                               double amount, String currency) {}
