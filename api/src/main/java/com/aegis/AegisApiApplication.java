package com.aegis;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/** AEGIS backend-for-frontend: orchestrates the FastAPI inference service, caps graphs for the
 *  frontend, and persists datasets/explanations in PostgreSQL (spec §8.3). */
@SpringBootApplication
public class AegisApiApplication {
    public static void main(String[] args) {
        SpringApplication.run(AegisApiApplication.class, args);
    }
}
