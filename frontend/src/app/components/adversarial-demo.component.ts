import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../services/api.service';
import { AdversarialArtifact } from '../models/api.models';

/** The showpiece: structural attack fools the naïve model, the hardened model holds (spec §7.8). */
@Component({
  selector: 'app-adversarial-demo',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="card">
      <h3>Adversarial robustness <span class="muted" style="font-weight:500">· naïve fooled vs hardened holds</span></h3>
      <p class="muted" style="font-size:13px;margin:6px 0 14px">A structural evasion attack rewires edges to hide laundering.
        The naïve model collapses; the self-loop-hardened model holds.</p>
      <button (click)="run()" [disabled]="loading">
        <span *ngIf="loading" class="spinner" style="width:13px;height:13px;border-width:2px;vertical-align:-2px"></span>
        {{ loading ? ' Running…' : 'Run before / after' }}
      </button>
      <p class="muted" *ngIf="error" style="margin-top:10px;color:var(--warn)">{{ error }}</p>
      <div *ngIf="art as a" style="margin-top:16px">
        <p style="margin:0 0 12px">{{ a.summary }}</p>
        <div class="metric-row">
          <div class="metric-card">
            <div class="metric" style="color:var(--illicit)">{{ pct(a.degradation.naive_attack_success_rate) }}</div>
            <div class="metric-label">naïve flags evaded</div>
          </div>
          <div class="metric-card">
            <div class="metric" style="color:var(--ok)">{{ pct(a.degradation.hardened_attack_success_rate) }}</div>
            <div class="metric-label">hardened flags evaded</div>
          </div>
          <div class="metric-card">
            <div class="metric grad">{{ pct(a.degradation.target_robustness_gap) }}</div>
            <div class="metric-label">robustness gap</div>
          </div>
        </div>
      </div>
    </div>`,
})
export class AdversarialDemoComponent {
  art: AdversarialArtifact | null = null;
  loading = false;
  error = '';
  constructor(private api: ApiService) {}
  run() {
    this.loading = true; this.error = '';
    this.api.adversarial().subscribe({
      next: (a) => { this.art = a; this.loading = false; },
      error: () => { this.error = 'No adversarial artifact yet — generate one with `python -m ml.adversarial …`.';
                     this.loading = false; },
    });
  }
  pct(v: number) { return (v * 100).toFixed(0) + '%'; }
}
