package com.aegis.controller;

import com.aegis.entity.AuditRecord;
import com.aegis.service.AuditService;
import java.time.Instant;
import java.util.List;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

/** Read access to the tamper-evident audit trail. Listing a subject's trail is open to any analyst;
 *  running a full chain verification is an admin action. */
@RestController
@RequestMapping("/api/audit")
@PreAuthorize("hasAnyRole('ANALYST','REVIEWER','ADMIN')")
public class AuditController {

    private final AuditService audit;

    public AuditController(AuditService audit) { this.audit = audit; }

    /** Recompute the whole hash chain and report integrity (and the first break, if any). */
    @GetMapping("/verify")
    @PreAuthorize("hasRole('ADMIN')")
    public AuditService.Verification verify() {
        return audit.verify();
    }

    /** The audit trail for one subject, e.g. ?subjectType=CASE&subjectId=42. */
    @GetMapping
    public List<RecordView> forSubject(@RequestParam String subjectType, @RequestParam Long subjectId) {
        return audit.forSubject(subjectType, subjectId).stream().map(RecordView::of).toList();
    }

    public record RecordView(long seq, Instant at, String actor, String action, String subjectType,
                             Long subjectId, String payload, String prevHash, String hash) {
        static RecordView of(AuditRecord r) {
            return new RecordView(r.getSeq(), r.getAt(), r.getActor(), r.getAction(), r.getSubjectType(),
                    r.getSubjectId(), r.getPayload(), r.getPrevHash(), r.getHash());
        }
    }
}
