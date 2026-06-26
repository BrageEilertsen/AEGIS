package com.aegis.service;

import com.aegis.domain.CaseEvent;
import com.aegis.domain.CaseState;
import com.aegis.domain.Disposition;
import com.aegis.domain.Priority;
import com.aegis.entity.Alert;
import com.aegis.entity.CaseActivity;
import com.aegis.entity.CaseFile;
import com.aegis.exception.NotFoundException;
import com.aegis.repository.AlertRepository;
import com.aegis.repository.CaseActivityRepository;
import com.aegis.repository.CaseRepository;
import com.aegis.workflow.CaseWorkflow;
import java.time.Instant;
import java.util.List;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/** Orchestrates the AML case lifecycle: triage alerts into cases, assign analysts, drive workflow
 *  transitions (validated by {@link CaseWorkflow}), and record every action. SLA windows are set when
 *  an investigation starts; dispositions are captured on close. */
@Service
public class CaseService {

    private final CaseRepository cases;
    private final AlertRepository alerts;
    private final CaseActivityRepository activities;
    private final CaseWorkflow workflow;
    private final AuditService audit;
    private final com.fasterxml.jackson.databind.ObjectMapper json;

    public CaseService(CaseRepository cases, AlertRepository alerts,
                       CaseActivityRepository activities, CaseWorkflow workflow,
                       AuditService audit, com.fasterxml.jackson.databind.ObjectMapper json) {
        this.cases = cases; this.alerts = alerts; this.activities = activities; this.workflow = workflow;
        this.audit = audit; this.json = json;
    }

    // ---- alerts ----
    @Transactional
    public Alert createAlert(Long datasetId, int nodeId, double score, String typology) {
        return alerts.findByDatasetIdAndNodeId(datasetId, nodeId)
                .orElseGet(() -> alerts.save(new Alert(datasetId, nodeId, score, typology)));
    }

    @Transactional(readOnly = true)
    public List<Alert> openAlerts(int limit) {
        return alerts.findByCaseIdIsNullAndDismissedFalse(PageRequest.of(0, Math.min(limit, 500)));
    }

    // ---- cases ----
    @Transactional
    public CaseFile createCase(String title, Priority priority, List<Long> alertIds, String actor) {
        CaseFile c = cases.save(new CaseFile(title, priority));
        if (alertIds != null) {
            for (Long aid : alertIds) {
                Alert a = alerts.findById(aid).orElseThrow(() -> new NotFoundException("alert " + aid));
                a.setCaseId(c.getId());
                alerts.save(a);
            }
        }
        log(c.getId(), actor, "CREATED", null, CaseState.NEW, title);
        audit.append(actor, "CASE_CREATED", "CASE", c.getId(),
                payload("title", title, "priority", c.getPriority(), "alertIds", alertIds));
        return c;
    }

    @Transactional
    public CaseFile assign(Long caseId, String assignee, String actor) {
        CaseFile c = require(caseId);
        CaseState from = c.getState();
        c.setState(workflow.apply(from, CaseEvent.ASSIGN));
        c.setAssignee(assignee);
        c.touch();
        cases.save(c);
        log(caseId, actor, "ASSIGN", from, c.getState(), "assigned to " + assignee);
        audit.append(actor, "CASE_ASSIGNED", "CASE", caseId,
                payload("assignee", assignee, "from", from, "to", c.getState()));
        return c;
    }

    /** Apply a workflow event with its side effects (SLA start, disposition on close). */
    @Transactional
    public CaseFile transition(Long caseId, CaseEvent event, String actor, String note) {
        CaseFile c = require(caseId);
        CaseState from = c.getState();
        CaseState to = workflow.apply(from, event);   // throws 409 if illegal from `from`
        c.setState(to);
        c.touch();
        switch (event) {
            case START_INVESTIGATION -> c.setSlaDueAt(Instant.now().plus(c.getPriority().sla()));
            case CLOSE_SAR -> close(c, Disposition.SAR_FILED, note, actor);
            case CLOSE_FALSE_POSITIVE -> close(c, Disposition.FALSE_POSITIVE, note, actor);
            case CLOSE_NO_ACTION -> close(c, Disposition.NO_ACTION, note, actor);
            default -> { /* ASSIGN handled in assign(); others have no side effect */ }
        }
        cases.save(c);
        log(caseId, actor, event.name(), from, to, note);
        audit.append(actor, "CASE_" + event.name(), "CASE", caseId,
                payload("from", from, "to", to, "note", note, "disposition", c.getDisposition()));
        return c;
    }

    /** Build a small JSON payload for the audit snapshot (keys/values; values may be null). */
    private String payload(Object... kv) {
        var map = new java.util.LinkedHashMap<String, Object>();
        for (int i = 0; i + 1 < kv.length; i += 2) map.put(String.valueOf(kv[i]), kv[i + 1]);
        try {
            return json.writeValueAsString(map);
        } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
            return map.toString();
        }
    }

    private void close(CaseFile c, Disposition d, String rationale, String actor) {
        c.setDisposition(d);
        c.setDispositionRationale(rationale);
        c.setClosedAt(Instant.now());
        c.setClosedBy(actor);
    }

    // ---- queries ----
    @Transactional(readOnly = true)
    public List<CaseFile> list(CaseState state, String assignee, int limit) {
        Pageable page = PageRequest.of(0, Math.min(limit, 500));
        if (state != null && assignee != null) return cases.findByStateAndAssignee(state, assignee, page);
        if (state != null) return cases.findByState(state, page);
        if (assignee != null) return cases.findByAssignee(assignee, page);
        return cases.findAll(page).getContent();
    }

    @Transactional(readOnly = true)
    public CaseFile require(Long caseId) {
        return cases.findById(caseId).orElseThrow(() -> new NotFoundException("case " + caseId));
    }

    @Transactional(readOnly = true)
    public List<Alert> alertsFor(Long caseId) { return alerts.findByCaseId(caseId); }

    @Transactional(readOnly = true)
    public List<CaseActivity> activitiesFor(Long caseId) {
        return activities.findByCaseIdOrderByAtAsc(caseId);
    }

    private void log(Long caseId, String actor, String action, CaseState from, CaseState to, String note) {
        activities.save(new CaseActivity(caseId, actor, action, from, to, note));
    }
}
