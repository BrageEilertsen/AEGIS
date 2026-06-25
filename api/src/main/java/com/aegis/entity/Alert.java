package com.aegis.entity;

import jakarta.persistence.*;
import java.time.Instant;

/** A flagged transaction surfaced for review. Raw signal; may be triaged into a {@link CaseFile}. */
@Entity
@Table(name = "alert", indexes = {
        @Index(name = "idx_alert_case", columnList = "caseId"),
        @Index(name = "idx_alert_node", columnList = "datasetId,nodeId")
})
public class Alert {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false) private Long datasetId;
    @Column(nullable = false) private int nodeId;
    @Column(nullable = false) private double score;
    private String typology;
    @Column(nullable = false) private Instant createdAt = Instant.now();
    private boolean dismissed = false;
    private Long caseId;            // null until triaged into a case

    protected Alert() {}

    public Alert(Long datasetId, int nodeId, double score, String typology) {
        this.datasetId = datasetId; this.nodeId = nodeId; this.score = score; this.typology = typology;
    }

    public Long getId() { return id; }
    public Long getDatasetId() { return datasetId; }
    public int getNodeId() { return nodeId; }
    public double getScore() { return score; }
    public String getTypology() { return typology; }
    public Instant getCreatedAt() { return createdAt; }
    public boolean isDismissed() { return dismissed; }
    public void setDismissed(boolean dismissed) { this.dismissed = dismissed; }
    public Long getCaseId() { return caseId; }
    public void setCaseId(Long caseId) { this.caseId = caseId; }
}
