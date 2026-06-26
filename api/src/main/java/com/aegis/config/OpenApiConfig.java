package com.aegis.config;

import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.info.License;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/** OpenAPI metadata for the Swagger UI (served at /swagger-ui.html, spec at /v3/api-docs). */
@Configuration
public class OpenApiConfig {

    @Bean
    OpenAPI aegisOpenApi() {
        return new OpenAPI().info(new Info()
                .title("AEGIS API")
                .version("1.0")
                .description("BFF for the AEGIS money-laundering detection platform: flagged "
                        + "transactions, faithful explanations, real-time monitoring, the AML "
                        + "case-management workflow, and the tamper-evident audit trail.")
                .license(new License().name("Portfolio project")));
    }
}
