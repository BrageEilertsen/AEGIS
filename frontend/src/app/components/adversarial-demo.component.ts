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
      <h3>⚔️ Adversarial robustness — naïve fooled vs hardened holds</h3>
      <button (click)="run()" [disabled]="loading">{{ loading ? 'Running…' : 'Run before/after' }}</button>
      <p class="muted" *ngIf="error">{{ error }}</p>
      <div *ngIf="art as a" style="margin-top:12px">
        <p>{{ a.summary }}</p>
        <div style="display:flex;gap:28px">
          <div><div class="metric" style="color:var(--illicit)">{{ pct(a.degradation.naive_attack_success_rate) }}</div>
               <div class="muted">naïve flags evaded</div></div>
          <div><div class="metric" style="color:var(--ok)">{{ pct(a.degradation.hardened_attack_success_rate) }}</div>
               <div class="muted">hardened flags evaded</div></div>
          <div><div class="metric">{{ pct(a.degradation.target_robustness_gap) }}</div>
               <div class="muted">robustness gap</div></div>
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
