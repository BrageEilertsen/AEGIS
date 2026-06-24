package com.aegis.exception;

/** The downstream FastAPI inference service failed or is unreachable. */
public class InferenceUnavailableException extends RuntimeException {
    public InferenceUnavailableException(String message, Throwable cause) { super(message, cause); }
}
