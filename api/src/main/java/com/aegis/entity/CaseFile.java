package com.aegis.entity;

import com.aegis.domain.CaseState;
import com.aegis.domain.Disposition;
import com.aegis.domain.Priority;
import jakarta.persistence.*;
import java.time.Instant;

/** An AML investigation case. State is governed by the workflow state machine; the entity is the
 *  persistent record of where the case is, who owns it, its SLA and (once closed) its disposition. */
@Entity
@Table(name = "case_file", indexes = {
        @Index(name = "idx_case_state", columnList = "state"),
        @Index(name = "idx_case_assignee", columnList = "assignee")
})
public class CaseFile {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false) private String title;

    @Enumerated(EnumType.STRING) @Column(nullable = false)
    private CaseState state = CaseState.NEW;

    @Enumerated(EnumType.STRING) @Column(nullable = false)
    private Priority priority = Priority.MEDIUM;

    private String assignee;                 // analyst principal name, null until assigned
    @Column(nullable = false) private Instant createdAt = Instant.now();
    @Column(nullable = false) private Instant updatedAt = Instant.now();
    private Instant slaDueAt;                 // set when investigation starts

    @Enumerated(EnumType.STRING) private Disposition disposition;   // set on close
    @Column(length = 2000) private String dispositionRationale;
    private Instant closedAt;
    private String closedBy;

    protected CaseFile() {}

    public CaseFile(String title, Priority priority) {
        this.title = title;
        if (priority != null) this.priority = priority;
    }

    /** Human-friendly reference derived from the id, e.g. CASE-000042. */
    @Transient
    public String getReference() { return id == null ? null : String.format("CASE-%06d", id); }

    public void touch() { this.updatedAt = Instant.now(); }
    public boolean isSlaBreached() {
        return slaDueAt != null && !state.isClosed() && Instant.now().isAfter(slaDueAt);
    }

    public Long getId() { return id; }
    public String getTitle() { return title; }
    public void setTitle(String title) { this.title = title; }
    public CaseState getState() { return state; }
    public void setState(CaseState state) { this.state = state; }
    public Priority getPriority() { return priority; }
    public void setPriority(Priority priority) { this.priority = priority; }
    public String getAssignee() { return assignee; }
    public void setAssignee(String assignee) { this.assignee = assignee; }
    public Instant getCreatedAt() { return createdAt; }
    public Instant getUpdatedAt() { return updatedAt; }
    public Instant getSlaDueAt() { return slaDueAt; }
    public void setSlaDueAt(Instant slaDueAt) { this.slaDueAt = slaDueAt; }
    public Disposition getDisposition() { return disposition; }
    public void setDisposition(Disposition disposition) { this.disposition = disposition; }
    public String getDispositionRationale() { return dispositionRationale; }
    public void setDispositionRationale(String r) { this.dispositionRationale = r; }
    public Instant getClosedAt() { return closedAt; }
    public void setClosedAt(Instant closedAt) { this.closedAt = closedAt; }
    public String getClosedBy() { return closedBy; }
    public void setClosedBy(String closedBy) { this.closedBy = closedBy; }
}
