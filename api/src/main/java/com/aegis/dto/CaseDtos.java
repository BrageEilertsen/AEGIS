package com.aegis.dto;

import com.aegis.domain.CaseState;
import com.aegis.domain.Disposition;
import com.aegis.domain.Priority;
import com.aegis.entity.Alert;
import com.aegis.entity.CaseActivity;
import com.aegis.entity.CaseFile;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.time.Instant;
import java.util.List;

/** Request/response shapes for the case-management API (records keep them immutable + concise). */
public final class CaseDtos {
    private CaseDtos() {}

    // ---- responses ----
    public record AlertView(Long id, Long datasetId, int nodeId, double score, String typology,
                            Instant createdAt, Long caseId) {
        public static AlertView of(Alert a) {
            return new AlertView(a.getId(), a.getDatasetId(), a.getNodeId(), a.getScore(),
                    a.getTypology(), a.getCreatedAt(), a.getCaseId());
        }
    }

    public record ActivityView(Instant at, String actor, String action,
                               CaseState fromState, CaseState toState, String note) {
        public static ActivityView of(CaseActivity x) {
            return new ActivityView(x.getAt(), x.getActor(), x.getAction(),
                    x.getFromState(), x.getToState(), x.getNote());
        }
    }

    public record CaseSummary(Long id, String reference, String title, CaseState state,
                              Priority priority, String assignee, Instant createdAt, Instant updatedAt,
                              Instant slaDueAt, boolean slaBreached, Disposition disposition) {
        public static CaseSummary of(CaseFile c) {
            return new CaseSummary(c.getId(), c.getReference(), c.getTitle(), c.getState(),
                    c.getPriority(), c.getAssignee(), c.getCreatedAt(), c.getUpdatedAt(),
                    c.getSlaDueAt(), c.isSlaBreached(), c.getDisposition());
        }
    }

    public record CaseDetail(CaseSummary summary, String dispositionRationale, String closedBy,
                             Instant closedAt, List<AlertView> alerts, List<ActivityView> activities) {
        public static CaseDetail of(CaseFile c, List<Alert> alerts, List<CaseActivity> activities) {
            return new CaseDetail(CaseSummary.of(c), c.getDispositionRationale(), c.getClosedBy(),
                    c.getClosedAt(), alerts.stream().map(AlertView::of).toList(),
                    activities.stream().map(ActivityView::of).toList());
        }
    }

    // ---- requests ----
    public record CreateAlertRequest(@NotNull Long datasetId, int nodeId, double score, String typology) {}

    public record CreateCaseRequest(@NotBlank String title, Priority priority, List<Long> alertIds) {}

    public record AssignRequest(@NotBlank String assignee) {}

    public record TransitionRequest(String note) {}

    public record DispositionRequest(@NotNull Disposition disposition, String rationale) {}
}
