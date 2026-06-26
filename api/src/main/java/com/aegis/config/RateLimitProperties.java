package com.aegis.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

/** Per-client rate-limit settings for the public API (cost/abuse guard). Two token buckets: a
 *  general one for all of {@code /api/**}, and a stricter one for the expensive
 *  explain/summary paths (CPU-heavy GNNExplainer + paid hosted LLM). */
@ConfigurationProperties(prefix = "aegis.ratelimit")
public record RateLimitProperties(boolean enabled, int capacity, int refillSeconds,
                                  int expensiveCapacity, int expensiveRefillSeconds) {
    public RateLimitProperties {
        if (capacity <= 0) capacity = 60;
        if (refillSeconds <= 0) refillSeconds = 60;
        if (expensiveCapacity <= 0) expensiveCapacity = 10;
        if (expensiveRefillSeconds <= 0) expensiveRefillSeconds = 60;
    }
}
