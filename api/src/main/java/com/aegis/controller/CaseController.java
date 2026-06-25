package com.aegis.controller;

import com.aegis.domain.CaseEvent;
import com.aegis.domain.CaseState;
import com.aegis.dto.CaseDtos.*;
import com.aegis.entity.CaseFile;
import com.aegis.service.CaseService;
import jakarta.validation.Valid;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import java.util.List;
import org.springframework.http.HttpStatus;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

/** AML case-management API. Requires a logged-in analyst (Entra ID); closures are restricted to
 *  reviewers/admins (segregation of duties — the analyst who investigates can't sign off their own
 *  SAR). The case lifecycle itself is enforced by the workflow state machine. */
@RestController
@RequestMapping("/api")
@Validated
@PreAuthorize("hasAnyRole('ANALYST','REVIEWER','ADMIN')")   // class-wide: must be a known user
public class CaseController {

    private final CaseService service;

    public CaseController(CaseService service) { this.service = service; }

    // ---- alerts ----
    @PostMapping("/alerts")
    @ResponseStatus(HttpStatus.CREATED)
    public AlertView createAlert(@Valid @RequestBody CreateAlertRequest req) {
        return AlertView.of(service.createAlert(req.datasetId(), req.nodeId(), req.score(), req.typology()));
    }

    @GetMapping("/alerts")
    public List<AlertView> openAlerts(@RequestParam(defaultValue = "100") @Min(1) @Max(500) int limit) {
        return service.openAlerts(limit).stream().map(AlertView::of).toList();
    }

    // ---- cases ----
    @PostMapping("/cases")
    @ResponseStatus(HttpStatus.CREATED)
    public CaseSummary create(@Valid @RequestBody CreateCaseRequest req) {
        CaseFile c = service.createCase(req.title(), req.priority(), req.alertIds(), actor());
        return CaseSummary.of(c);
    }

    @GetMapping("/cases")
    public List<CaseSummary> list(@RequestParam(required = false) CaseState state,
                                  @RequestParam(required = false) String assignee,
                                  @RequestParam(defaultValue = "100") @Min(1) @Max(500) int limit) {
        return service.list(state, assignee, limit).stream().map(CaseSummary::of).toList();
    }

    @GetMapping("/cases/{id}")
    public CaseDetail get(@PathVariable @Min(1) Long id) {
        CaseFile c = service.require(id);
        return CaseDetail.of(c, service.alertsFor(id), service.activitiesFor(id));
    }

    @PostMapping("/cases/{id}/assign")
    public CaseSummary assign(@PathVariable @Min(1) Long id, @Valid @RequestBody AssignRequest req) {
        return CaseSummary.of(service.assign(id, req.assignee(), actor()));
    }

    @PostMapping("/cases/{id}/start")
    public CaseSummary start(@PathVariable @Min(1) Long id, @RequestBody(required = false) TransitionRequest req) {
        return CaseSummary.of(service.transition(id, CaseEvent.START_INVESTIGATION, actor(), note(req)));
    }

    @PostMapping("/cases/{id}/submit")
    public CaseSummary submit(@PathVariable @Min(1) Long id, @RequestBody(required = false) TransitionRequest req) {
        return CaseSummary.of(service.transition(id, CaseEvent.SUBMIT_FOR_REVIEW, actor(), note(req)));
    }

    @PostMapping("/cases/{id}/return")
    @PreAuthorize("hasAnyRole('REVIEWER','ADMIN')")
    public CaseSummary returnForRework(@PathVariable @Min(1) Long id,
                                       @RequestBody(required = false) TransitionRequest req) {
        return CaseSummary.of(service.transition(id, CaseEvent.RETURN_FOR_REWORK, actor(), note(req)));
    }

    /** Close the case with a disposition — reviewers/admins only. */
    @PostMapping("/cases/{id}/disposition")
    @PreAuthorize("hasAnyRole('REVIEWER','ADMIN')")
    public CaseSummary dispose(@PathVariable @Min(1) Long id, @Valid @RequestBody DispositionRequest req) {
        return CaseSummary.of(service.transition(id, req.disposition().event(), actor(), req.rationale()));
    }

    private static String note(TransitionRequest req) { return req == null ? null : req.note(); }

    private static String actor() {
        Authentication a = SecurityContextHolder.getContext().getAuthentication();
        return a != null ? a.getName() : "system";
    }
}
