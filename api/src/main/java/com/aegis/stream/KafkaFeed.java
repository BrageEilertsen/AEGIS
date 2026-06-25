package com.aegis.stream;

import jakarta.annotation.PostConstruct;
import org.springframework.boot.autoconfigure.condition.ConditionalOnExpression;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

/** Real transaction feed: consumes {@link TransactionEvent} JSON from a Kafka topic and drives the
 *  same {@link StreamProcessor} as the simulated feed. Active only when AEGIS_KAFKA_BROKERS is set
 *  (e.g. an Azure Event Hubs Kafka endpoint or a Kafka/Redpanda container); otherwise the simulated
 *  feed runs and no broker connection is attempted. */
@Component
@ConditionalOnExpression("!'${aegis.kafka.brokers:}'.isEmpty()")
public class KafkaFeed {

    private final StreamProcessor processor;

    public KafkaFeed(StreamProcessor processor) { this.processor = processor; }

    @PostConstruct
    void init() { processor.setSource("kafka"); }

    @KafkaListener(topics = "${aegis.kafka.topic:transactions}", groupId = "${aegis.kafka.group:aegis-stream}")
    public void onMessage(TransactionEvent event) {
        processor.process(event);
    }
}
