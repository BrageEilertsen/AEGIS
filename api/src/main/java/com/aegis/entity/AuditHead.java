package com.aegis.entity;

import jakarta.persistence.*;

/** Single-row pointer to the tail of the audit chain (last seq + last hash). Appends take a
 *  pessimistic lock on this row so the chain stays strictly ordered and fork-free even with
 *  concurrent writers / multiple API replicas. */
@Entity
@Table(name = "audit_head")
public class AuditHead {
    @Id private Long id;            // always 1
    private long lastSeq;
    @Column(length = 64) private String lastHash;

    protected AuditHead() {}

    public AuditHead(Long id, long lastSeq, String lastHash) {
        this.id = id; this.lastSeq = lastSeq; this.lastHash = lastHash;
    }

    public Long getId() { return id; }
    public long getLastSeq() { return lastSeq; }
    public void setLastSeq(long lastSeq) { this.lastSeq = lastSeq; }
    public String getLastHash() { return lastHash; }
    public void setLastHash(String lastHash) { this.lastHash = lastHash; }
}
