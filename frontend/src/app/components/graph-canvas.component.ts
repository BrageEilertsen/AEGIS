import { AfterViewInit, Component, ElementRef, Input, OnChanges, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import cytoscape from 'cytoscape';
import { NeighborhoodSubgraph } from '../models/api.models';

/** Renders the capped flagged subgraph + neighbourhood (spec §8.4 — never the full graph).
 *  Illicit nodes red, the flagged target outlined, score gradient otherwise; force-directed layout
 *  so laundering shapes (fan-out, chain, cycle) read at a glance. */
@Component({
  selector: 'app-graph-canvas',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div style="position:relative">
      <div #cy style="height:420px;border:1px solid var(--border);border-radius:12px;
                      background:radial-gradient(circle at 50% 40%, #0e1626 0%, #0a0f1a 100%)"></div>
      <div *ngIf="(subgraph?.node_ids?.length ?? 0) <= 1"
           style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
                  pointer-events:none;text-align:center">
        <div class="faint" style="font-size:13px">Isolated transaction — no connected neighbourhood to draw.<br>
          <span style="font-size:12px">(its own features drive the score; see the panel below)</span></div>
      </div>
    </div>
    <div class="muted" style="font-size:12px;margin-top:10px;display:flex;gap:8px;flex-wrap:wrap">
      <span class="pill" style="background:rgba(255,93,108,.15);border-color:rgba(255,93,108,.4);color:var(--illicit)">● flagged target</span>
      <span class="pill" style="background:rgba(245,166,35,.15);border-color:rgba(245,166,35,.4);color:var(--warn)">● known illicit</span>
      <span class="pill">● licit (shade = score)</span>
    </div>`,
})
export class GraphCanvasComponent implements AfterViewInit, OnChanges {
  @Input() subgraph: NeighborhoodSubgraph | null = null;
  @ViewChild('cy', { static: true }) cyRef!: ElementRef<HTMLDivElement>;
  private cy?: cytoscape.Core;

  ngAfterViewInit() { this.render(); }
  ngOnChanges() { this.render(); }

  private render() {
    const sg = this.subgraph;
    if (!this.cyRef || !sg) return;
    const ids = sg.node_ids;
    const nodes = ids.map((nid, i) => ({
      data: { id: String(nid), label: String(nid),
              kind: nid === sg.target_node_id ? 'target' : (sg.node_labels[i] === 1 ? 'illicit' : 'licit'),
              score: sg.node_scores[i] ?? 0 },
    }));
    const [src, dst] = sg.edge_index?.length ? sg.edge_index : [[], []];
    const edges = src.map((s, e) => ({ data: { id: `e${e}`, source: String(ids[s]), target: String(ids[dst[e]]) } }));
    this.cy?.destroy();
    this.cy = cytoscape({
      container: this.cyRef.nativeElement,
      elements: [...nodes, ...edges],
      style: [
        { selector: 'node', style: { 'background-color': '#3d4d6b', label: 'data(label)',
            color: '#aebdd4', 'font-size': '7px', 'font-family': 'ui-monospace, monospace',
            width: 18, height: 18, 'border-width': 1, 'border-color': '#1a2334' } },
        { selector: 'node[kind="target"]', style: { 'background-color': '#ff5d6c',
            'border-width': 3, 'border-color': '#fff', width: 30, height: 30 } },
        { selector: 'node[kind="illicit"]', style: { 'background-color': '#f5a623' } },
        { selector: 'edge', style: { width: 1.6, 'line-color': '#46577a',
            'target-arrow-color': '#46577a', 'target-arrow-shape': 'triangle',
            'curve-style': 'bezier', 'arrow-scale': 0.8 } },
      ],
      layout: { name: 'cose', animate: false, padding: 24 },
    });
  }
}
