package com.aegis.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

/** CORS for the Angular UI. Defaults to allowing ANY origin (this is a public, read-only demo API
 *  with no cookies/credentials), so the deployed frontend works with zero configuration. Set
 *  AEGIS_CORS_ORIGINS to a comma-separated list to restrict it. */
@Configuration
public class WebConfig implements WebMvcConfigurer {

    @Value("${AEGIS_CORS_ORIGINS:*}")
    private String allowedOrigins;

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        var mapping = registry.addMapping("/api/**")
                .allowedMethods("GET", "POST", "OPTIONS")
                .allowedHeaders("*");
        if (allowedOrigins == null || allowedOrigins.isBlank() || allowedOrigins.trim().equals("*")) {
            mapping.allowedOriginPatterns("*");   // any origin (no credentials → safe + simple)
        } else {
            mapping.allowedOrigins(allowedOrigins.split("\\s*,\\s*"));
        }
    }
}
