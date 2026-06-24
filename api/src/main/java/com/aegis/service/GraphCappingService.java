package com.aegis.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.util.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

/**
 * Owns the graph-capping logic (spec §8.3/§8.4): the frontend must never receive an unrenderable
 * graph. Given an explanation's neighbourhood subgraph, keep the target node plus the highest-score
 * neighbours up to {@code maxNodes}, drop edges whose endpoints are not both kept, and relabel.
 */
@Service
public class GraphCappingService {

    private final ObjectMapper mapper;
    private final int maxNodes;

    public GraphCappingService(ObjectMapper mapper,
                               @Value("${aegis.graph.max-nodes:300}") int maxNodes) {
        this.mapper = mapper;
        this.maxNodes = maxNodes;
    }

    /** Return a copy of the explanation with its neighbourhood_subgraph capped to maxNodes. */
    public JsonNode capExplanation(JsonNode explanation) {
        JsonNode sg = explanation.get("neighborhood_subgraph");
        if (sg == null || !sg.has("node_ids")) return explanation;
        ArrayNode nodeIds = (ArrayNode) sg.get("node_ids");
        if (nodeIds.size() <= maxNodes) return explanation;

        int target = explanation.path("node_id").asInt();
        ArrayNode scores = (ArrayNode) sg.get("node_scores");
        // keep the target's index + the top (maxNodes-1) indices by score
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < nodeIds.size(); i++) order.add(i);
        order.sort((a, b) -> Double.compare(scores.get(b).asDouble(), scores.get(a).asDouble()));
        LinkedHashSet<Integer> keep = new LinkedHashSet<>();
        for (int i = 0; i < nodeIds.size(); i++) if (nodeIds.get(i).asInt() == target) keep.add(i);
        for (int idx : order) { if (keep.size() >= maxNodes) break; keep.add(idx); }
        List<Integer> kept = new ArrayList<>(keep);
        Map<Integer, Integer> remap = new HashMap<>();
        for (int n = 0; n < kept.size(); n++) remap.put(kept.get(n), n);

        ObjectNode cappedSg = sg.deepCopy();
        cappedSg.set("node_ids", subset(nodeIds, kept));
        cappedSg.set("node_labels", subset((ArrayNode) sg.get("node_labels"), kept));
        cappedSg.set("node_scores", subset((ArrayNode) sg.get("node_scores"), kept));
        // remap edges
        ArrayNode ei = (ArrayNode) sg.get("edge_index");
        ArrayNode imp = (ArrayNode) sg.get("edge_importance");
        ArrayNode src = ei.size() > 0 ? (ArrayNode) ei.get(0) : mapper.createArrayNode();
        ArrayNode dst = ei.size() > 1 ? (ArrayNode) ei.get(1) : mapper.createArrayNode();
        ArrayNode ns = mapper.createArrayNode(), nd = mapper.createArrayNode(), ni = mapper.createArrayNode();
        for (int e = 0; e < src.size(); e++) {
            int s = src.get(e).asInt(), d = dst.get(e).asInt();
            if (remap.containsKey(s) && remap.containsKey(d)) {
                ns.add(remap.get(s)); nd.add(remap.get(d));
                if (imp != null && e < imp.size()) ni.add(imp.get(e));
            }
        }
        ArrayNode newEi = mapper.createArrayNode(); newEi.add(ns); newEi.add(nd);
        cappedSg.set("edge_index", newEi);
        cappedSg.set("edge_importance", ni);
        cappedSg.put("was_capped", true);

        ObjectNode out = explanation.deepCopy();
        out.set("neighborhood_subgraph", cappedSg);
        return out;
    }

    private ArrayNode subset(ArrayNode arr, List<Integer> idx) {
        ArrayNode out = mapper.createArrayNode();
        for (int i : idx) out.add(arr.get(i));
        return out;
    }
}
