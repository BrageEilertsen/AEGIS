import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
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

      <div class="ai-summary" *ngIf="ex.summary">
        <div class="ai-summary-label">AI summary <span>· generated from the evidence below, grounded — not free-form</span></div>
        <p>{{ ex.summary }}</p>
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
export class ExplanationPanelComponent {
  @Input() explanation: Explanation | null = null;
  bar(v: number, ex: Explanation) {
    const max = Math.max(...ex.top_features.map((f) => f.importance), 1e-6);
    return Math.round((v / max) * 100);
  }
}
