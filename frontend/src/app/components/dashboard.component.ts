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
    <div *ngIf="dataset as d" style="display:flex;align-items:center;gap:10px;margin-bottom:18px;flex-wrap:wrap">
      <span class="pill">{{ d.name }}</span>
      <span class="muted" style="font-size:13px">{{ d.numNodes | number }} transactions ·
        <span style="color:var(--illicit);font-weight:600">{{ d.numIllicit | number }} illicit</span>
        ({{ (d.illicitRatio * 100).toFixed(3) }}%)</span>
    </div>

    <app-metrics-panel [metrics]="metrics" style="display:block;margin-bottom:18px"></app-metrics-panel>

    <div class="grid-main">
      <div class="card">
        <h3>Flagged transactions</h3>
        <p class="muted" style="font-size:12px;margin:6px 0 12px">Ranked by illicit probability · click one to explain it</p>
        <div class="table-scroll">
          <table>
            <thead><tr><th>node</th><th>score</th><th>label</th></tr></thead>
            <tbody>
              <tr *ngFor="let f of flags" (click)="select(f)" [class.selected]="f === selected">
                <td class="mono">#{{ f.node_id }}</td>
                <td><span class="score-chip" [class.hot]="f.score >= 0.9">{{ f.score.toFixed(3) }}</span></td>
                <td>
                  <span class="badge badge-illicit" *ngIf="f.label === 1">illicit</span>
                  <span class="badge badge-licit" *ngIf="f.label !== 1">licit</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div>
        <div class="card" style="margin-bottom:18px" *ngIf="selected">
          <h3>Responsible neighbourhood</h3>
          <div *ngIf="explanation; else computing" style="margin-top:12px">
            <app-graph-canvas [subgraph]="explanation.neighborhood_subgraph"></app-graph-canvas>
          </div>
          <ng-template #computing>
            <div class="loading-pane" *ngIf="!explainError; else explainFailed">
              <span class="spinner"></span>
              <div>
                <div style="color:var(--ink);font-weight:600">Computing explanation for #{{ selected.node_id }}…</div>
                <div class="faint" style="font-size:12px;margin-top:4px">GNNExplainer is finding the minimal sufficient subgraph
                  — usually about a second.</div>
              </div>
            </div>
            <ng-template #explainFailed>
              <div class="loading-pane">
                <div>
                  <div style="color:var(--illicit);font-weight:600">Couldn't load the explanation for #{{ selected.node_id }}.</div>
                  <div class="faint" style="font-size:12px;margin-top:4px">The model service may be waking up — click the
                    transaction again in a moment.</div>
                </div>
              </div>
            </ng-template>
          </ng-template>
        </div>
        <app-explanation-panel [explanation]="explanation" [datasetId]="dataset?.id ?? null"></app-explanation-panel>
        <div class="card" *ngIf="!selected" style="text-align:center;color:var(--muted)">
          ← Select a flagged transaction to see its faithful explanation.
        </div>
      </div>
    </div>

    <div style="margin-top:18px"><app-adversarial-demo></app-adversarial-demo></div>
  `,
})
export class DashboardComponent implements OnInit {
  dataset: Dataset | null = null;
  metrics: Metrics | null = null;
  flags: Flag[] = [];
  selected: Flag | null = null;
  explanation: Explanation | null = null;
  explainError = false;

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
    const id = this.dataset.id;
    this.selected = f; this.explanation = null; this.explainError = false;
    // investigate = explanation + AI summary fetched concurrently; if the summary isn't ready inline,
    // the panel polls /api/summary. If investigate hiccups (e.g. mid-deploy), fall back to plain
    // explain so the UI never hangs on "Computing…".
    this.api.investigate(id, f.node_id).subscribe({
      next: (e) => (this.explanation = e),
      error: () => this.api.explain(id, f.node_id).subscribe({
        next: (e) => (this.explanation = e),
        error: () => { if (this.selected === f) this.explainError = true; },
      }),
    });
  }
}
