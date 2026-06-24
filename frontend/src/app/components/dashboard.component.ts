import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../services/api.service';
import { Dataset, Explanation, Flag, Metrics } from '../models/api.models';
import { MetricsPanelComponent } from './metrics-panel.component';
import { GraphCanvasComponent } from './graph-canvas.component';
import { ExplanationPanelComponent } from './explanation-panel.component';
import { AdversarialDemoComponent } from './adversarial-demo.component';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, MetricsPanelComponent, GraphCanvasComponent,
            ExplanationPanelComponent, AdversarialDemoComponent],
  template: `
    <div *ngIf="dataset as d" style="margin-bottom:14px">
      <span class="pill">{{ d.name }}</span>
      <span class="muted"> {{ d.numNodes | number }} transactions · {{ d.numIllicit | number }} illicit
        ({{ (d.illicitRatio * 100).toFixed(3) }}%)</span>
    </div>

    <app-metrics-panel [metrics]="metrics" style="display:block;margin-bottom:16px"></app-metrics-panel>

    <div style="display:grid;grid-template-columns:340px 1fr;gap:16px;align-items:start">
      <div class="card">
        <h3>🚩 Flagged transactions</h3>
        <table>
          <thead><tr><th>node</th><th>score</th><th>label</th></tr></thead>
          <tbody>
            <tr *ngFor="let f of flags" (click)="select(f)">
              <td>#{{ f.node_id }}</td><td>{{ f.score.toFixed(3) }}</td>
              <td>{{ f.label === 1 ? '⚠️ illicit' : 'licit' }}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div>
        <div class="card" style="margin-bottom:16px" *ngIf="selected">
          <h3>Responsible neighbourhood {{ explanation ? '' : '(loading…)' }}</h3>
          <app-graph-canvas [subgraph]="explanation?.neighborhood_subgraph ?? null"></app-graph-canvas>
        </div>
        <app-explanation-panel [explanation]="explanation"></app-explanation-panel>
        <p class="muted" *ngIf="!selected">Select a flagged transaction to see its faithful explanation.</p>
      </div>
    </div>

    <div style="margin-top:16px"><app-adversarial-demo></app-adversarial-demo></div>
  `,
})
export class DashboardComponent implements OnInit {
  dataset: Dataset | null = null;
  metrics: Metrics | null = null;
  flags: Flag[] = [];
  selected: Flag | null = null;
  explanation: Explanation | null = null;

  constructor(private api: ApiService) {}

  ngOnInit() {
    this.api.datasets().subscribe((ds) => {
      this.dataset = ds[0] ?? null;
      if (this.dataset) {
        this.api.metrics(this.dataset.id).subscribe((m) => (this.metrics = m));
        this.api.flags(this.dataset.id, 0.5, 100).subscribe((f) => (this.flags = f));
      }
    });
  }

  select(f: Flag) {
    if (!this.dataset) return;
    this.selected = f; this.explanation = null;
    this.api.explain(this.dataset.id, f.node_id).subscribe((e) => (this.explanation = e));
  }
}
