// TypeScript views of the API payloads. Flags/metrics/explanation/adversarial are forwarded
// verbatim from the Python contracts (snake_case); datasets come from Java records (camelCase).

export interface Dataset {
  id: number; name: string; variant: string;
  numNodes: number; numEdges: number; numIllicit: number; illicitRatio: number;
}

export interface Flag { node_id: number; score: number; label: number; }

export interface Metrics {
  pr_auc: number | null; roc_auc: number | null;
  recall_at_precision: number | null; min_precision: number | null;
  f1_illicit: number | null; n_total: number; n_pos: number;
  confusion_matrix: { tn: number; fp: number; fn: number; tp: number } | null;
}

export interface TopEdge { source_node: number; target_node: number; importance: number; source: string; }
export interface TopFeature { column_name: string; feature_index: number; importance: number; group: string; }
export interface Typology {
  label: string; confidence: number; justification: string;
  scores: Record<string, number>; ground_truth: string | null;
}
export interface NeighborhoodSubgraph {
  target_node_id: number; node_ids: number[]; edge_index: number[][];
  node_labels: number[]; node_scores: number[]; edge_importance: number[];
  was_capped: boolean; num_hops: number;
}
export interface Explanation {
  schema_version: string; node_id: number; score: number; predicted_label: number;
  top_edges: TopEdge[]; top_features: TopFeature[]; matched_typology: Typology;
  neighborhood_subgraph: NeighborhoodSubgraph;
  faithfulness: { method: string; edge_importance_source: string; note: string };
  model_version: string;
}

export interface AdversarialArtifact {
  schema_version: string; summary: string;
  degradation: {
    naive_attack_success_rate: number; hardened_attack_success_rate: number;
    target_robustness_gap: number; naive_mean_score_drop: number; hardened_mean_score_drop: number;
    naive_pr_auc_drop: number; hardened_pr_auc_drop: number;
  };
  per_target: Array<Record<string, any>>;
  metrics: Record<string, any>;
}
