import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Metrics } from '../models/api.models';

@Component({
  selector: 'app-metrics-panel',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="card" *ngIf="metrics as m">
      <h3>Held-out test performance <span class="muted">(illicit class)</span></h3>
      <div style="display:flex;gap:28px;flex-wrap:wrap">
        <div><div class="metric">{{ fmt(m.pr_auc) }}</div><div class="muted">PR-AUC</div></div>
        <div><div class="metric">{{ fmt(m.roc_auc) }}</div><div class="muted">ROC-AUC</div></div>
        <div><div class="metric">{{ fmt(m.recall_at_precision) }}</div>
             <div class="muted">recall @ p≥{{ m.min_precision }}</div></div>
        <div><div class="metric">{{ fmt(m.f1_illicit) }}</div><div class="muted">F1 (illicit)</div></div>
      </div>
      <p class="muted" style="font-size:12px">Accuracy is meaningless at ~{{ pct(m.n_pos, m.n_total) }} positives —
        PR-AUC and recall-at-precision are the headline (spec §7.6).</p>
    </div>`,
})
export class MetricsPanelComponent {
  @Input() metrics: Metrics | null = null;
  fmt(v: number | null) { return v == null ? '—' : v.toFixed(3); }
  pct(a: number, b: number) { return b ? ((100 * a) / b).toFixed(1) + '%' : '—'; }
}
