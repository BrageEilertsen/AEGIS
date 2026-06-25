package com.aegis.exception;

import jakarta.validation.ConstraintViolationException;
import java.net.URI;
import java.time.Instant;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ProblemDetail;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;

/** Maps exceptions to RFC 7807 {@link ProblemDetail} responses (application/problem+json), so
 *  clients get a consistent, machine-readable error shape. */
@RestControllerAdvice
public class GlobalExceptionHandler {

    private static final Logger log = LoggerFactory.getLogger(GlobalExceptionHandler.class);

    @ExceptionHandler(NotFoundException.class)
    public ProblemDetail notFound(NotFoundException e) {
        return problem(HttpStatus.NOT_FOUND, "Not Found", e.getMessage());
    }

    @ExceptionHandler(InferenceUnavailableException.class)
    public ProblemDetail inference(InferenceUnavailableException e) {
        return problem(HttpStatus.BAD_GATEWAY, "Inference Unavailable", e.getMessage());
    }

    @ExceptionHandler({MethodArgumentNotValidException.class, ConstraintViolationException.class,
                       MethodArgumentTypeMismatchException.class})
    public ProblemDetail invalid(Exception e) {
        return problem(HttpStatus.BAD_REQUEST, "Validation Failed", e.getMessage());
    }

    @ExceptionHandler(Exception.class)
    public ProblemDetail unexpected(Exception e) {
        log.error("Unhandled exception", e);
        return problem(HttpStatus.INTERNAL_SERVER_ERROR, "Internal Server Error",
                "An unexpected error occurred.");   // don't leak internals to clients
    }

    private ProblemDetail problem(HttpStatus status, String title, String detail) {
        ProblemDetail pd = ProblemDetail.forStatusAndDetail(status, detail == null ? title : detail);
        pd.setTitle(title);
        pd.setType(URI.create("https://aegis/errors/" + status.value()));
        pd.setProperty("timestamp", Instant.now().toString());
        return pd;
    }
}
