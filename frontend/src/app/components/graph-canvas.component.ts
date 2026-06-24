import { AfterViewInit, Component, ElementRef, Input, OnChanges, ViewChild } from '@angular/core';
import cytoscape from 'cytoscape';
import { NeighborhoodSubgraph } from '../models/api.models';

/** Renders the capped flagged subgraph + neighbourhood (spec §8.4 — never the full graph).
 *  Illicit nodes red, the flagged target outlined, score gradient otherwise; force-directed layout
 *  so laundering shapes (fan-out, chain, cycle) read at a glance. */
@Component({
  selector: 'app-graph-canvas',
  standalone: true,
  template: `<div #cy style="height:420px;border:1px solid #283142;border-radius:8px"></div>
    <div class="muted" style="font-size:12px;margin-top:6px">
      <span class="pill" style="background:#e74c3c">flagged target</span>
      <span class="pill" style="background:#e67e22">known illicit</span>
      <span class="pill">licit (shade = score)</span></div>`,
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
        { selector: 'node', style: { 'background-color': '#3a4a63', label: 'data(label)',
            color: '#cdd9e5', 'font-size': '7px', width: 16, height: 16 } },
        { selector: 'node[kind="target"]', style: { 'background-color': '#e74c3c',
            'border-width': 3, 'border-color': '#fff', width: 26, height: 26 } },
        { selector: 'node[kind="illicit"]', style: { 'background-color': '#e67e22' } },
        { selector: 'edge', style: { width: 1.5, 'line-color': '#5a6b82',
            'target-arrow-color': '#5a6b82', 'target-arrow-shape': 'triangle',
            'curve-style': 'bezier', 'arrow-scale': 0.7 } },
      ],
      layout: { name: 'cose', animate: false, padding: 20 },
    });
  }
}
