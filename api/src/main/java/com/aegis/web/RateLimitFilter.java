package com.aegis.web;

import com.aegis.config.RateLimitProperties;
import io.github.bucket4j.Bandwidth;
import io.github.bucket4j.Bucket;
import io.github.bucket4j.ConsumptionProbe;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.time.Duration;
import java.util.concurrent.ConcurrentHashMap;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.core.annotation.Order;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

/** Per-client token-bucket rate limiting for {@code /api/**}, so the public demo can't be used to
 *  run up CPU (GNNExplainer) or hosted-LLM cost. Two buckets per client: a general one for every
 *  API call and a stricter one for the expensive explain/summary paths (a request to those consumes
 *  from both). Client identity is the first {@code X-Forwarded-For} hop (Azure Container Apps sets
 *  it) falling back to the socket address. In-memory per instance — fine for this scale; a
 *  distributed store (Redis/Hazelcast) is the swap-in if it ever scales out. */
@Component
@Order(1)
@EnableConfigurationProperties(RateLimitProperties.class)
public class RateLimitFilter extends OncePerRequestFilter {

    private final RateLimitProperties props;
    private final ConcurrentHashMap<String, Bucket> general = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Bucket> expensive = new ConcurrentHashMap<>();

    public RateLimitFilter(RateLimitProperties props) {
        this.props = props;
    }

    @Override
    protected boolean shouldNotFilter(HttpServletRequest req) {
        // Don't rate-limit CORS preflight (no cost) or non-API paths.
        return !props.enabled() || "OPTIONS".equalsIgnoreCase(req.getMethod())
                || !req.getRequestURI().startsWith("/api/");
    }

    @Override
    protected void doFilterInternal(HttpServletRequest req, HttpServletResponse res, FilterChain chain)
            throws ServletException, IOException {
        String client = clientKey(req);
        String uri = req.getRequestURI();

        // Expensive endpoints consume from BOTH buckets (stricter limit dominates).
        if (isExpensive(uri)) {
            ConsumptionProbe ex = bucket(expensive, client, props.expensiveCapacity(), props.expensiveRefillSeconds())
                    .tryConsumeAndReturnRemaining(1);
            if (!ex.isConsumed()) { reject(res, ex.getNanosToWaitForRefill()); return; }
        }
        ConsumptionProbe gen = bucket(general, client, props.capacity(), props.refillSeconds())
                .tryConsumeAndReturnRemaining(1);
        if (!gen.isConsumed()) { reject(res, gen.getNanosToWaitForRefill()); return; }

        res.setHeader("X-RateLimit-Remaining", Long.toString(gen.getRemainingTokens()));
        chain.doFilter(req, res);
    }

    private static boolean isExpensive(String uri) {
        return uri.startsWith("/api/explain/") || uri.startsWith("/api/summary/")
                || uri.startsWith("/api/investigate/") || uri.startsWith("/api/graph/")
                || uri.startsWith("/api/adversarial");
    }

    private Bucket bucket(ConcurrentHashMap<String, Bucket> store, String key, int capacity, int seconds) {
        return store.computeIfAbsent(key, k -> Bucket.builder()
                .addLimit(Bandwidth.builder().capacity(capacity)
                        .refillGreedy(capacity, Duration.ofSeconds(seconds)).build())
                .build());
    }

    private static String clientKey(HttpServletRequest req) {
        String xff = req.getHeader("X-Forwarded-For");
        if (xff != null && !xff.isBlank()) return xff.split(",")[0].trim();
        return req.getRemoteAddr();
    }

    private void reject(HttpServletResponse res, long nanosToWait) throws IOException {
        long retryAfter = Math.max(1, nanosToWait / 1_000_000_000L);
        res.setStatus(HttpStatus.TOO_MANY_REQUESTS.value());
        res.setContentType(MediaType.APPLICATION_PROBLEM_JSON_VALUE);
        res.setHeader("Retry-After", Long.toString(retryAfter));
        // This filter runs before Spring MVC CORS, so add the header ourselves or the browser hides
        // the 429 behind a CORS error (read-only public demo → any origin is fine).
        res.setHeader("Access-Control-Allow-Origin", "*");
        res.getWriter().write("""
            {"type":"about:blank","title":"Too Many Requests","status":429,\
            "detail":"Rate limit exceeded. Retry in %d seconds.","retryAfterSeconds":%d}"""
                .formatted(retryAfter, retryAfter));
    }
}
