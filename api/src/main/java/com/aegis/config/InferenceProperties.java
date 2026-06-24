package com.aegis.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

/** Config for the downstream Python inference service. */
@ConfigurationProperties(prefix = "aegis.inference")
public record InferenceProperties(String baseUrl, int timeoutSeconds) {
    public InferenceProperties {
        if (baseUrl == null || baseUrl.isBlank()) baseUrl = "http://localhost:8000";
        if (timeoutSeconds <= 0) timeoutSeconds = 600;
    }
}
