package com.aegis.stream;

import jakarta.annotation.PostConstruct;
import java.time.Instant;
import java.util.concurrent.ThreadLocalRandom;
import org.springframework.boot.autoconfigure.condition.ConditionalOnExpression;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

/** Default transaction feed for the live demo: emits synthetic transactions among a pool of
 *  accounts, periodically injecting a laundering motif (a fan-out burst, or an outsized "high-value"
 *  transfer) so the monitoring engine visibly fires alerts. Active unless a real Kafka broker is
 *  configured (AEGIS_KAFKA_BROKERS), in which case {@link KafkaFeed} takes over. */
@Component
@ConditionalOnExpression("'${aegis.kafka.brokers:}'.isEmpty()")
public class SimulatedFeed {

    private final StreamProcessor processor;
    private final StreamProperties props;
    private String[] accounts;
    private long counter;
    private String burstSource;
    private int burstRemaining;

    public SimulatedFeed(StreamProcessor processor, StreamProperties props) {
        this.processor = processor; this.props = props;
    }

    @PostConstruct
    void init() {
        accounts = new String[props.accounts()];
        for (int i = 0; i < accounts.length; i++) accounts[i] = String.format("ACC%04d", i + 1);
        processor.setSource("simulated");
    }

    @Scheduled(fixedRateString = "${aegis.stream.emitIntervalMs:700}", initialDelay = 3000)
    void tick() {
        if (!props.enabled()) return;
        processor.process(nextEvent());
    }

    private TransactionEvent nextEvent() {
        ThreadLocalRandom rnd = ThreadLocalRandom.current();
        counter++;
        String source, target;
        double amount;

        if (burstRemaining > 0) {                 // mid fan-out burst: one source -> many targets
            burstRemaining--;
            source = burstSource;
            target = pick(rnd, source);
            amount = rnd.nextDouble(200, 1500);
        } else if (counter % 23 == 0) {           // start a new fan-out burst
            burstSource = accounts[rnd.nextInt(accounts.length)];
            burstRemaining = 8;
            source = burstSource;
            target = pick(rnd, source);
            amount = rnd.nextDouble(200, 1500);
        } else if (counter % 37 == 0) {           // an outsized high-value transfer
            source = accounts[rnd.nextInt(accounts.length)];
            target = pick(rnd, source);
            amount = rnd.nextDouble(40_000, 120_000);
        } else {                                  // ordinary background traffic
            source = accounts[rnd.nextInt(accounts.length)];
            target = pick(rnd, source);
            amount = Math.exp(rnd.nextGaussian() * 0.8 + 6);   // log-normal-ish, ~e^6 ≈ 400
        }
        return new TransactionEvent(String.format("TX-%08d", counter), Instant.now(),
                source, target, Math.round(amount * 100) / 100.0, "USD");
    }

    private String pick(ThreadLocalRandom rnd, String not) {
        String a;
        do { a = accounts[rnd.nextInt(accounts.length)]; } while (a.equals(not));
        return a;
    }
}
