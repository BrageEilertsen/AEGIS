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
      <p><span class="metric">{{ (ex.score * 100).toFixed(1) }}%</span>
         <span class="muted"> illicit probability · predicted {{ ex.predicted_label === 1 ? 'ILLICIT' : 'licit' }}</span></p>

      <h4>Matched laundering typology</h4>
      <p><span class="pill" style="background:#e67e22;color:#1a1a1a">{{ ex.matched_typology.label }}</span>
         <span class="muted"> confidence {{ ex.matched_typology.confidence }}</span></p>
      <p class="muted" style="font-size:13px">{{ ex.matched_typology.justification }}</p>

      <h4>Top contributing features</h4>
      <div *ngFor="let f of ex.top_features" style="margin:3px 0">
        <span style="display:inline-block;width:160px;font-size:12px">{{ f.column_name }}</span>
        <span style="display:inline-block;height:10px;background:#3fb6ff;vertical-align:middle"
              [style.width.px]="bar(f.importance, ex)"></span>
        <span class="muted" style="font-size:11px"> {{ f.importance.toFixed(3) }} · {{ f.group }}</span>
      </div>

      <h4>Most influential edges</h4>
      <ul style="font-size:12px;margin:4px 0">
        <li *ngFor="let e of ex.top_edges.slice(0, 6)">
          #{{ e.source_node }} → #{{ e.target_node }} <span class="muted">({{ e.importance.toFixed(2) }})</span></li>
      </ul>

      <p class="muted" style="font-size:11px">Edge importance via <b>{{ ex.faithfulness.edge_importance_source }}</b>.
        {{ ex.faithfulness.note }}</p>
    </div>`,
})
export class ExplanationPanelComponent {
  @Input() explanation: Explanation | null = null;
  bar(v: number, ex: Explanation) {
    const max = Math.max(...ex.top_features.map((f) => f.importance), 1e-6);
    return Math.round((v / max) * 180);
  }
}
