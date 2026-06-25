import { Component, Input, OnChanges, OnDestroy, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subscription, timer } from 'rxjs';
import { ApiService } from '../services/api.service';
import { Explanation } from '../models/api.models';

@Component({
  selector: 'app-explanation-panel',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="card" *ngIf="explanation as ex">
      <h3>Why transaction #{{ ex.node_id }} was flagged</h3>
      <div style="display:flex;align-items:baseline;gap:10px;margin-top:10px">
        <span class="metric grad">{{ (ex.score * 100).toFixed(1) }}%</span>
        <span class="muted">illicit probability ·
          <span class="badge" [class.badge-illicit]="ex.predicted_label === 1" [class.badge-licit]="ex.predicted_label !== 1">
            predicted {{ ex.predicted_label === 1 ? 'ILLICIT' : 'licit' }}</span>
        </span>
      </div>

      <div class="ai-summary">
        <div class="ai-summary-label">
          <ng-container *ngIf="aiReady">AI summary <span>· written by an LLM, grounded in the evidence below</span></ng-container>
          <ng-container *ngIf="!aiReady && loading">AI summary <span>· writing<span class="dots"></span></span></ng-container>
          <ng-container *ngIf="!aiReady && !loading">Summary <span>· grounded summary of the evidence below</span></ng-container>
        </div>
        <p *ngIf="summaryText">{{ summaryText }}</p>
        <p *ngIf="!summaryText && loading" class="skel" style="height:34px"></p>
      </div>

      <h4>Matched laundering typology</h4>
      <p style="margin:0">
        <span class="pill" style="background:rgba(245,166,35,.15);border-color:rgba(245,166,35,.4);color:var(--warn)">
          {{ ex.matched_typology.label }}</span>
        <span class="muted" style="font-size:12px"> · confidence {{ ex.matched_typology.confidence }}</span>
      </p>
      <p class="muted" style="font-size:13px;margin:8px 0 0">{{ ex.matched_typology.justification }}</p>

      <h4>Top contributing features</h4>
      <div class="feat-row" *ngFor="let f of ex.top_features">
        <span class="feat-name" [title]="f.column_name">{{ f.column_name }}</span>
        <span class="feat-track"><span class="feat-fill" [style.width.%]="bar(f.importance, ex)"></span></span>
        <span class="feat-val">{{ f.importance.toFixed(3) }}</span>
      </div>

      <h4>Most influential edges</h4>
      <ul style="font-size:13px;margin:4px 0;padding-left:18px;color:var(--muted)">
        <li *ngFor="let e of ex.top_edges.slice(0, 6)">
          <span class="mono" style="color:var(--ink)">#{{ e.source_node }} → #{{ e.target_node }}</span>
          <span class="faint"> ({{ e.importance.toFixed(2) }})</span></li>
        <li *ngIf="!ex.top_edges.length" class="faint">isolated transaction — no incident edges</li>
      </ul>

      <hr class="div">
      <p class="faint" style="font-size:11px;margin:0">Edge importance via <b style="color:var(--muted)">{{ ex.faithfulness.edge_importance_source }}</b>.
        {{ ex.faithfulness.note }}</p>
    </div>`,
})
export class ExplanationPanelComponent implements OnChanges, OnDestroy {
  @Input() explanation: Explanation | null = null;
  @Input() datasetId: number | null = null;

  summaryText = '';
  aiReady = false;     // the LLM summary has arrived
  loading = false;     // waiting on the LLM (show a "writing…" state, not the template)
  private sub?: Subscription;

  constructor(private api: ApiService) {}

  ngOnChanges(_: SimpleChanges) {
    this.sub?.unsubscribe();
    this.aiReady = false;
    const ex = this.explanation;

    // No LLM upgrade coming (disabled): show the grounded template immediately.
    if (!ex || this.datasetId == null || !ex.summary_pending) {
      this.loading = false;
      this.summaryText = ex?.summary ?? '';
      return;
    }

    // LLM is generating: show a brief "writing…" state (NOT the template), then the LLM text.
    // The LLM is hosted and fast (~1-3s); poll every 1.5s and, if it never arrives within ~45s,
    // fall back to the grounded template so the box is never empty.
    this.loading = true;
    this.summaryText = '';
    const node = ex.node_id;
    const fallback = ex.summary ?? '';
    this.sub = timer(1200, 1500).subscribe((i) => {
      this.api.summary(this.datasetId!, node).subscribe({
        next: (s) => {
          if (this.explanation?.node_id !== node) { this.stop(); return; }
          if (s.ready && s.summary) {
            this.summaryText = s.summary; this.aiReady = true; this.stop();
          } else if (s.ready || i >= 28) {
            this.summaryText = fallback; this.stop();   // LLM off/empty or ~45s elapsed
          }
        },
        error: () => { this.summaryText = fallback; this.stop(); },
      });
    });
  }

  private stop() { this.loading = false; this.sub?.unsubscribe(); }
  ngOnDestroy() { this.sub?.unsubscribe(); }

  bar(v: number, ex: Explanation) {
    const max = Math.max(...ex.top_features.map((f) => f.importance), 1e-6);
    return Math.round((v / max) * 100);
  }
}
