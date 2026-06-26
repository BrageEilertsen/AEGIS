package com.aegis.stream;

import com.aegis.service.CaseService;
import java.time.Instant;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

/** The real-time monitoring engine. Maintains a sliding time-window of recent transactions as an
 *  in-memory account graph, scores each arriving transaction against that window (a cheap structural
 *  pre-screen — fan-in / fan-out / velocity / amount anomaly, the laundering motifs), pushes every
 *  transaction to subscribers, and raises an Alert (persisted so it enters the analyst queue) when a
 *  transaction crosses the risk threshold. High-risk hits are exactly what a GNN deep-dive (the
 *  /explain path) would then be run on — streaming pre-screen, batch deep analysis. */
@Component
public class StreamProcessor {

    private static final Logger log = LoggerFactory.getLogger(StreamProcessor.class);
    private static final double FANOUT_REF = 6, FANIN_REF = 6, VELOCITY_REF = 12;

    private final StreamProperties props;
    private final StreamBroadcaster broadcaster;
    private final CaseService cases;

    private final Deque<TransactionEvent> window = new ArrayDeque<>();
    private final AtomicLong totalTx = new AtomicLong();
    private final AtomicLong totalAlerts = new AtomicLong();
    private volatile String source = "simulated";

    public StreamProcessor(StreamProperties props, StreamBroadcaster broadcaster, CaseService cases) {
        this.props = props; this.broadcaster = broadcaster; this.cases = cases;
    }

    public void setSource(String source) { this.source = source; }

    /** Ingest one transaction: window it, score it, broadcast it, and alert if risky. */
    public void process(TransactionEvent e) {
        Score s;
        synchronized (window) {
            window.addLast(e);
            evictOld(e.ts());
            s = score(e);
        }
        totalTx.incrementAndGet();
        boolean flagged = s.risk() >= props.alertThreshold();
        broadcaster.publish("tx", new StreamTx(e.id(), e.ts().toEpochMilli(), e.source(), e.target(),
                round(e.amount()), round(s.risk()), flagged, s.pattern()));
        if (flagged) {
            totalAlerts.incrementAndGet();
            raiseAlert(s.account(), s.pattern(), s.risk(), e.ts());
        }
    }

    private void raiseAlert(String account, String pattern, double risk, Instant ts) {
        broadcaster.publish("alert", new StreamAlert(ts.toEpochMilli(), account, pattern, round(risk)));
        try {   // persist so it shows up in the analyst alert queue (datasetId 0 = stream origin)
            cases.createAlert(0L, Math.abs(account.hashCode()), risk, pattern + " · " + account);
        } catch (RuntimeException ex) {
            log.debug("stream alert persist failed (non-fatal): {}", ex.toString());
        }
    }

    private void evictOld(Instant now) {
        long cutoff = now.toEpochMilli() - props.windowSeconds() * 1000L;
        while (!window.isEmpty() && window.peekFirst().ts().toEpochMilli() < cutoff) {
            window.pollFirst();
        }
    }

    /** Structural risk of a transaction given the current window. */
    private Score score(TransactionEvent e) {
        Set<String> outTargets = new HashSet<>(), inSources = new HashSet<>();
        int srcVelocity = 0, dstVelocity = 0;
        double sum = 0, sumSq = 0;
        int n = 0;
        for (TransactionEvent w : window) {
            if (w.source().equals(e.source())) { outTargets.add(w.target()); srcVelocity++; }
            if (w.target().equals(e.target())) { inSources.add(w.source()); }
            if (w.source().equals(e.target()) || w.target().equals(e.target())) { dstVelocity++; }
            sum += w.amount(); sumSq += w.amount() * w.amount(); n++;
        }
        double fanOut = Math.min(1, outTargets.size() / FANOUT_REF);
        double fanIn = Math.min(1, inSources.size() / FANIN_REF);
        double velocity = Math.min(1, Math.max(srcVelocity, dstVelocity) / VELOCITY_REF);
        double amountZ = amountZ(e.amount(), sum, sumSq, n);
        double amount = Math.min(1, Math.max(0, (amountZ - 2) / 2));   // only unusually large amounts

        // Max-of-signals (so any single strong laundering motif fires) nudged by velocity.
        double dominant = Math.max(Math.max(fanOut, fanIn), amount);
        double risk = clamp(0.85 * dominant + 0.15 * velocity);

        String pattern;
        String account;
        if (fanOut >= fanIn && fanOut >= amount) { pattern = "fan-out"; account = e.source(); }
        else if (fanIn >= amount) { pattern = "fan-in"; account = e.target(); }
        else { pattern = "high-value"; account = e.source(); }
        return new Score(risk, pattern, account);
    }

    private static double amountZ(double x, double sum, double sumSq, int n) {
        if (n < 2) return 0;
        double mean = sum / n;
        double var = Math.max(1e-9, sumSq / n - mean * mean);
        return (x - mean) / Math.sqrt(var);
    }

    public StreamStats stats() {
        int recent;
        synchronized (window) {
            long since = Instant.now().toEpochMilli() - 1000;
            recent = (int) window.stream().filter(w -> w.ts().toEpochMilli() >= since).count();
            return new StreamStats(totalTx.get(), totalAlerts.get(), window.size(), recent,
                    broadcaster.subscribers(), props.alertThreshold(), source);
        }
    }

    private static double clamp(double v) { return Math.max(0, Math.min(1, v)); }
    private static double round(double v) { return Math.round(v * 1000.0) / 1000.0; }

    private record Score(double risk, String pattern, String account) {}

    // --- broadcast payloads ---
    public record StreamTx(String id, long ts, String source, String target, double amount,
                           double risk, boolean flagged, String pattern) {}
    public record StreamAlert(long ts, String account, String pattern, double risk) {}
    public record StreamStats(long totalTransactions, long totalAlerts, int windowSize,
                              int throughputPerSec, int subscribers, double threshold, String source) {}
}
