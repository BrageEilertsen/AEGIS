package com.aegis.entity;

import jakarta.persistence.*;
import java.time.Instant;

/** One append-only entry in the tamper-evident audit chain. Each record stores the hash of the
 *  previous record, so the whole log forms a hash chain: altering any past record (or reordering /
 *  deleting one) breaks every subsequent hash, which {@code AuditService.verify()} detects. */
@Entity
@Table(name = "audit_record", indexes = {
        @Index(name = "idx_audit_seq", columnList = "seq", unique = true),
        @Index(name = "idx_audit_subject", columnList = "subjectType,subjectId")
})
public class AuditRecord {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, unique = true) private long seq;   // 1-based position in the chain
    @Column(nullable = false) private Instant at;                // for display
    @Column(nullable = false) private long atMillis;             // hashed (exact, round-trip-safe)
    @Column(nullable = false) private String actor;
    @Column(nullable = false) private String action;
    @Column(nullable = false) private String subjectType;
    private Long subjectId;
    @Column(columnDefinition = "text") private String payload;   // JSON snapshot of the decision
    @Column(nullable = false, length = 64) private String prevHash;
    @Column(nullable = false, length = 64) private String hash;

    protected AuditRecord() {}

    public AuditRecord(long seq, Instant at, String actor, String action, String subjectType,
                       Long subjectId, String payload, String prevHash, String hash) {
        this.seq = seq; this.at = at; this.atMillis = at.toEpochMilli();
        this.actor = actor; this.action = action;
        this.subjectType = subjectType; this.subjectId = subjectId; this.payload = payload;
        this.prevHash = prevHash; this.hash = hash;
    }

    public Long getId() { return id; }
    public long getSeq() { return seq; }
    public Instant getAt() { return at; }
    public long getAtMillis() { return atMillis; }
    public String getActor() { return actor; }
    public String getAction() { return action; }
    public String getSubjectType() { return subjectType; }
    public Long getSubjectId() { return subjectId; }
    public String getPayload() { return payload; }
    public String getPrevHash() { return prevHash; }
    public String getHash() { return hash; }
}
