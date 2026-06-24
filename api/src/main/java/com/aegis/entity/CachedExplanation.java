package com.aegis.entity;

import com.fasterxml.jackson.databind.JsonNode;
import io.hypersistence.utils.hibernate.type.json.JsonType;
import jakarta.persistence.*;
import org.hibernate.annotations.Type;

/** Cache of a node's explanation contract (stored verbatim as JSONB) to avoid recomputing the
 *  expensive GNNExplainer run on repeated views. */
@Entity
@Table(name = "cached_explanation",
       uniqueConstraints = @UniqueConstraint(columnNames = {"datasetId", "nodeId"}))
public class CachedExplanation {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    private Long datasetId;
    private int nodeId;

    @Type(JsonType.class)
    @Column(columnDefinition = "jsonb", nullable = false)
    private JsonNode payload;

    protected CachedExplanation() {}

    public CachedExplanation(Long datasetId, int nodeId, JsonNode payload) {
        this.datasetId = datasetId; this.nodeId = nodeId; this.payload = payload;
    }

    public JsonNode getPayload() { return payload; }
}
