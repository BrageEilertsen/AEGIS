import { Component, NgZone, OnDestroy, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subscription, timer } from 'rxjs';
import { ApiService } from '../services/api.service';
import { StreamAlert, StreamStats, StreamTx } from '../models/api.models';

/** Live transaction-monitoring view. Opens an SSE connection to the BFF and shows a real-time
 *  transaction ticker + alert feed + throughput stats — the streaming pre-screen in action. */
@Component({
  selector: 'app-live-monitor',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="card">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px">
        <h3 style="margin:0">Real-time monitoring
          <span class="live-dot" [class.off]="!connected"></span>
          <span class="muted" style="font-size:12px;font-weight:400">{{ connected ? 'live' : 'connecting…' }}</span>
        </h3>
        <span class="muted" style="font-size:12px" *ngIf="stats as s">
          source: <span class="mono" style="color:var(--ink)">{{ s.source }}</span></span>
      </div>
      <p class="muted" style="font-size:12px;margin:6px 0 12px">A simulated transaction stream is scored on arrival against a sliding
        time-window; structural laundering motifs (fan-out, fan-in, high-value) raise alerts that flow into the case queue.</p>

      <div class="stat-row" *ngIf="stats as s">
        <div class="stat"><div class="stat-num grad">{{ s.throughputPerSec }}</div><div class="stat-lbl">tx / sec</div></div>
        <div class="stat"><div class="stat-num">{{ s.totalTransactions | number }}</div><div class="stat-lbl">processed</div></div>
        <div class="stat"><div class="stat-num" style="color:var(--illicit)">{{ s.totalAlerts | number }}</div><div class="stat-lbl">alerts</div></div>
        <div class="stat"><div class="stat-num">{{ s.windowSize | number }}</div><div class="stat-lbl">in window</div></div>
      </div>

      <div class="monitor-grid">
        <div>
          <h4>Transaction feed</h4>
          <div class="feed">
            <div class="feed-row" *ngFor="let t of txs" [class.flagged]="t.flagged">
              <span class="mono faint">{{ t.id }}</span>
              <span class="mono">{{ t.source }} → {{ t.target }}</span>
              <span class="amt">{{ t.amount | currency:'USD':'symbol':'1.0-0' }}</span>
              <span class="risk-chip" [class.hot]="t.flagged">{{ (t.risk * 100).toFixed(0) }}</span>
            </div>
            <div *ngIf="!txs.length" class="faint" style="font-size:12px;padding:8px">waiting for transactions…</div>
          </div>
        </div>
        <div>
          <h4>Live alerts</h4>
          <div class="feed">
            <div class="alert-row" *ngFor="let a of alerts">
              <span class="pill" style="background:rgba(214,69,69,.14);border-color:rgba(214,69,69,.4);color:var(--illicit)">{{ a.pattern }}</span>
              <span class="mono" style="color:var(--ink)">{{ a.account }}</span>
              <span class="risk-chip hot">{{ (a.risk * 100).toFixed(0) }}</span>
            </div>
            <div *ngIf="!alerts.length" class="faint" style="font-size:12px;padding:8px">no alerts yet — a fan-out burst will trigger one shortly.</div>
          </div>
        </div>
      </div>
    </div>`,
  styles: [`
    .live-dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--ok,#3fb950);margin:0 4px 0 6px;box-shadow:0 0 0 0 rgba(63,185,80,.6);animation:pulse 1.6s infinite}
    .live-dot.off{background:var(--muted);animation:none}
    @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(63,185,80,.5)}70%{box-shadow:0 0 0 6px rgba(63,185,80,0)}100%{box-shadow:0 0 0 0 rgba(63,185,80,0)}}
    .stat-row{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}
    .stat{flex:1;min-width:90px;background:var(--panel,rgba(255,255,255,.03));border:1px solid var(--border,rgba(255,255,255,.08));border-radius:10px;padding:10px 12px}
    .stat-num{font-size:22px;font-weight:700;line-height:1}
    .stat-lbl{font-size:11px;color:var(--muted);margin-top:4px;text-transform:uppercase;letter-spacing:.05em}
    .monitor-grid{display:grid;grid-template-columns:1.4fr 1fr;gap:16px}
    @media(max-width:760px){.monitor-grid{grid-template-columns:1fr}}
    .feed{height:280px;overflow:hidden;border:1px solid var(--border,rgba(255,255,255,.08));border-radius:10px}
    .feed-row,.alert-row{display:flex;align-items:center;gap:8px;padding:7px 10px;font-size:12px;border-bottom:1px solid rgba(255,255,255,.05);animation:slidein .25s ease}
    .feed-row .amt{margin-left:auto;color:var(--muted)}
    .feed-row.flagged{background:rgba(214,69,69,.08)}
    .alert-row .risk-chip{margin-left:auto}
    @keyframes slidein{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:none}}
    .risk-chip{font-weight:700;font-size:11px;padding:1px 7px;border-radius:20px;background:rgba(255,255,255,.08);color:var(--muted)}
    .risk-chip.hot{background:rgba(214,69,69,.18);color:var(--illicit)}
  `],
})
export class LiveMonitorComponent implements OnInit, OnDestroy {
  txs: StreamTx[] = [];
  alerts: StreamAlert[] = [];
  stats: StreamStats | null = null;
  connected = false;

  private es?: EventSource;
  private statsSub?: Subscription;

  constructor(private api: ApiService, private zone: NgZone) {}

  ngOnInit() {
    this.connect();
    this.statsSub = timer(0, 2000).subscribe(() =>
      this.api.streamStats().subscribe({ next: (s) => (this.stats = s), error: () => {} }));
  }

  private connect() {
    // EventSource callbacks fire outside Angular's zone — re-enter so the view updates.
    this.es = new EventSource(this.api.streamUrl());
    this.es.onopen = () => this.zone.run(() => (this.connected = true));
    this.es.onerror = () => this.zone.run(() => (this.connected = false));
    this.es.addEventListener('tx', (e) => this.zone.run(() => {
      this.txs = [JSON.parse((e as MessageEvent).data), ...this.txs].slice(0, 18);
    }));
    this.es.addEventListener('alert', (e) => this.zone.run(() => {
      this.alerts = [JSON.parse((e as MessageEvent).data), ...this.alerts].slice(0, 8);
    }));
  }

  ngOnDestroy() {
    this.es?.close();
    this.statsSub?.unsubscribe();
  }
}
