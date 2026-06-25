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

      <div class="ai-summary" *ngIf="summaryText">
        <div class="ai-summary-label">
          {{ aiReady ? 'AI summary' : 'Summary' }}
          <span *ngIf="aiReady">· rephrased by a local LLM from the evidence below, grounded — not free-form</span>
          <span *ngIf="!aiReady && polling">· generating a plain-English summary<span class="dots"></span></span>
          <span *ngIf="!aiReady && !polling">· grounded summary of the evidence below</span>
        </div>
        <p>{{ summaryText }}</p>
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
  aiReady = false;     // the fluent LLM rephrasing has arrived
  polling = false;     // a background generation is being polled for
  private sub?: Subscription;

  constructor(private api: ApiService) {}

  ngOnChanges(_: SimpleChanges) {
    this.sub?.unsubscribe();
    this.aiReady = false;
    this.polling = false;
    const ex = this.explanation;
    this.summaryText = ex?.summary ?? '';
    if (!ex || this.datasetId == null || !ex.summary_pending) return;
    // The instant template summary already shows; poll the BFF until the LLM upgrade is ready
    // (every 2.5s, give up after ~30s and keep the template).
    this.polling = true;
    const node = ex.node_id;
    this.sub = timer(2500, 2500).subscribe((i) => {
      this.api.summary(this.datasetId!, node).subscribe({
        next: (s) => {
          if (this.explanation?.node_id !== node) { this.stopPolling(); return; }
          if (s.ready) {
            if (s.summary) { this.summaryText = s.summary; this.aiReady = true; }
            this.stopPolling();
          } else if (i >= 11) {
            this.stopPolling();   // ~30s elapsed — keep the grounded template
          }
        },
        error: () => this.stopPolling(),
      });
    });
  }

  private stopPolling() { this.polling = false; this.sub?.unsubscribe(); }
  ngOnDestroy() { this.sub?.unsubscribe(); }

  bar(v: number, ex: Explanation) {
    const max = Math.max(...ex.top_features.map((f) => f.importance), 1e-6);
    return Math.round((v / max) * 100);
  }
}
