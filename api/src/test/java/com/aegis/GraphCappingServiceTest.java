package com.aegis.service;

import static org.assertj.core.api.Assertions.assertThat;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;

class GraphCappingServiceTest {

    private final ObjectMapper mapper = new ObjectMapper();

    /** A 6-node neighbourhood capped to 3 must keep the target, drop to <=3 nodes, and only retain
     *  edges whose endpoints both survive (no dangling edges sent to the frontend). */
    @Test
    void capsToMaxNodesKeepingTargetAndConsistentEdges() {
        var svc = new GraphCappingService(mapper, 3);
        ObjectNode explanation = mapper.createObjectNode();
        explanation.put("node_id", 100);                 // target == node_ids[0]
        ObjectNode sg = explanation.putObject("neighborhood_subgraph");
        sg.put("target_node_id", 100);
        sg.set("node_ids", ints(100, 101, 102, 103, 104, 105));
        sg.set("node_labels", ints(1, 0, 0, 0, 0, 0));
        sg.set("node_scores", doubles(0.9, 0.8, 0.7, 0.6, 0.5, 0.4));
        ArrayNode ei = mapper.createArrayNode();
        ei.add(ints(0, 1, 4));                            // edges 0->1, 1->2, 4->5 (relabeled)
        ei.add(ints(1, 2, 5));
        sg.set("edge_index", ei);
        sg.set("edge_importance", doubles(0.5, 0.4, 0.3));

        var capped = svc.capExplanation(explanation).get("neighborhood_subgraph");
        assertThat(capped.get("node_ids")).hasSize(3);
        assertThat(capped.get("was_capped").asBoolean()).isTrue();
        // target retained
        boolean hasTarget = false;
        for (var n : capped.get("node_ids")) hasTarget |= n.asInt() == 100;
        assertThat(hasTarget).isTrue();
        // every retained edge references valid (in-range) relabeled node indices
        int n = capped.get("node_ids").size();
        var src = capped.get("edge_index").get(0);
        var dst = capped.get("edge_index").get(1);
        for (int e = 0; e < src.size(); e++) {
            assertThat(src.get(e).asInt()).isBetween(0, n - 1);
            assertThat(dst.get(e).asInt()).isBetween(0, n - 1);
        }
        assertThat(src.size()).isEqualTo(capped.get("edge_importance").size());
    }

    private ArrayNode ints(int... v) {
        ArrayNode a = mapper.createArrayNode();
        for (int x : v) a.add(x);
        return a;
    }

    private ArrayNode doubles(double... v) {
        ArrayNode a = mapper.createArrayNode();
        for (double x : v) a.add(x);
        return a;
    }
}
