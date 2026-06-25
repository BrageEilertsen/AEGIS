import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet],
  template: `
    <header class="app-header">
      <div class="logo"><svg width="18" height="18" viewBox="0 0 24 24" fill="#fff" aria-hidden="true"><path d="M12 2l8 3v6c0 5-3.4 8.6-8 11-4.6-2.4-8-6-8-11V5l8-3z"/></svg></div>
      <div>
        <div class="brand">AE<span class="accent">GIS</span></div>
        <div class="tagline">Money-laundering detection on transaction graphs · explainable · adversarially robust</div>
      </div>
      <span class="badge-live"><span class="status-dot"></span> live model</span>
    </header>
    <main style="padding:24px;max-width:1440px;margin:0 auto"><router-outlet></router-outlet></main>
  `,
})
export class AppComponent {}
