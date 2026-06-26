package com.aegis.stream;

import org.springframework.boot.context.properties.ConfigurationProperties;

/** Tunables for the real-time monitoring pipeline. */
@ConfigurationProperties(prefix = "aegis.stream")
public record StreamProperties(boolean enabled, int windowSeconds, int emitIntervalMs,
                               double alertThreshold, int accounts) {
    public StreamProperties {
        if (windowSeconds <= 0) windowSeconds = 90;
        if (emitIntervalMs <= 0) emitIntervalMs = 700;
        if (alertThreshold <= 0) alertThreshold = 0.75;
        if (accounts <= 0) accounts = 60;
    }
}
