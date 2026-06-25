package com.aegis.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

/** CORS for the Angular UI. Origins are configurable via AEGIS_CORS_ORIGINS (comma-separated) so the
 *  same build serves the local dev server and a deployed frontend. Use "*" to allow any origin
 *  (fine for this read-only demo API; no credentials/cookies are used). */
@Configuration
public class WebConfig implements WebMvcConfigurer {

    @Value("${aegis.cors.allowed-origins:http://localhost:4200,http://localhost}")
    private String[] allowedOrigins;

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        var mapping = registry.addMapping("/api/**").allowedMethods("GET", "POST");
        if (allowedOrigins.length == 1 && "*".equals(allowedOrigins[0])) {
            mapping.allowedOriginPatterns("*");
        } else {
            mapping.allowedOrigins(allowedOrigins);
        }
    }
}
