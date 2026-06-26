package com.aegis.stream;

import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.EnableScheduling;

/** Enables the scheduled simulated feed and binds the stream tunables. */
@Configuration
@EnableScheduling
@EnableConfigurationProperties(StreamProperties.class)
public class StreamConfig {}
