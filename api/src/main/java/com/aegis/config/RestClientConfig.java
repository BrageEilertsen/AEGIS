package com.aegis.config;

import java.time.Duration;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.web.client.RestClient;

/** Builds the RestClient used to call the FastAPI inference service. */
@Configuration
@EnableConfigurationProperties(InferenceProperties.class)
public class RestClientConfig {

    @Bean
    public RestClient inferenceRestClient(InferenceProperties props) {
        var factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(Duration.ofSeconds(10));
        factory.setReadTimeout(Duration.ofSeconds(props.timeoutSeconds()));
        return RestClient.builder().baseUrl(props.baseUrl()).requestFactory(factory).build();
    }
}
