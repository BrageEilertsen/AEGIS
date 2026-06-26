package com.aegis.exception;

/** Thrown when a case workflow event is not legal from the case's current state (maps to HTTP 409). */
public class IllegalCaseTransitionException extends RuntimeException {
    public IllegalCaseTransitionException(String message) { super(message); }
}
