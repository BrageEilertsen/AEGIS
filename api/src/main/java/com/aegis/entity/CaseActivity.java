package com.aegis.entity;

import com.aegis.domain.CaseState;
import jakarta.persistence.*;
import java.time.Instant;

/** An immutable entry in a case's action history (assignment, transition, disposition, note). This
 *  is the human-readable audit trail; Phase 5 adds the tamper-evident hash-chained record on top. */
@Entity
@Table(name = "case_activity", indexes = @Index(name = "idx_activity_case", columnList = "caseId,at"))
public class CaseActivity {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false) private Long caseId;
    @Column(nullable = false) private Instant at = Instant.now();
    @Column(nullable = false) private String actor;
    @Column(nullable = false) private String action;
    @Enumerated(EnumType.STRING) private CaseState fromState;
    @Enumerated(EnumType.STRING) private CaseState toState;
    @Column(length = 2000) private String note;

    protected CaseActivity() {}

    public CaseActivity(Long caseId, String actor, String action,
                        CaseState fromState, CaseState toState, String note) {
        this.caseId = caseId; this.actor = actor; this.action = action;
        this.fromState = fromState; this.toState = toState; this.note = note;
    }

    public Long getId() { return id; }
    public Long getCaseId() { return caseId; }
    public Instant getAt() { return at; }
    public String getActor() { return actor; }
    public String getAction() { return action; }
    public CaseState getFromState() { return fromState; }
    public CaseState getToState() { return toState; }
    public String getNote() { return note; }
}
