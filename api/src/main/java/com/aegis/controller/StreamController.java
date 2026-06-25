package com.aegis.controller;

import com.aegis.stream.StreamBroadcaster;
import com.aegis.stream.StreamProcessor;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

/** Public live-monitoring feed. The browser opens an SSE connection to {@code /api/stream} and
 *  receives {@code tx} and {@code alert} events in real time; {@code /api/stream/stats} gives a
 *  point-in-time snapshot (throughput, window size, totals). Read-only, so it stays on the public
 *  demo tier. */
@RestController
@RequestMapping("/api/stream")
public class StreamController {

    private final StreamBroadcaster broadcaster;
    private final StreamProcessor processor;

    public StreamController(StreamBroadcaster broadcaster, StreamProcessor processor) {
        this.broadcaster = broadcaster; this.processor = processor;
    }

    @GetMapping(produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter stream() {
        return broadcaster.register();
    }

    @GetMapping("/stats")
    public StreamProcessor.StreamStats stats() {
        return processor.stats();
    }
}
