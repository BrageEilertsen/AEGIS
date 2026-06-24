import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet],
  template: `
    <header style="padding:14px 22px;border-bottom:1px solid #283142;display:flex;gap:12px;align-items:baseline">
      <h1 style="margin:0">🛡️ AEGIS</h1>
      <span class="muted">money-laundering detection on transaction graphs · explainable · adversarially robust</span>
    </header>
    <main style="padding:22px;max-width:1400px;margin:0 auto"><router-outlet></router-outlet></main>
  `,
})
export class AppComponent {}
