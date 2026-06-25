package com.aegis.stream;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import org.springframework.stereotype.Component;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

/** Fans out live stream events (transactions + alerts) to all connected browsers over Server-Sent
 *  Events. Dead connections are pruned on the next publish. */
@Component
public class StreamBroadcaster {

    private final List<SseEmitter> emitters = new CopyOnWriteArrayList<>();

    public SseEmitter register() {
        SseEmitter emitter = new SseEmitter(0L);   // no timeout; the feed keeps it warm
        emitters.add(emitter);
        emitter.onCompletion(() -> emitters.remove(emitter));
        emitter.onTimeout(() -> emitters.remove(emitter));
        emitter.onError(e -> emitters.remove(emitter));
        return emitter;
    }

    public int subscribers() { return emitters.size(); }

    public void publish(String event, Object data) {
        for (SseEmitter emitter : emitters) {
            try {
                emitter.send(SseEmitter.event().name(event).data(data));
            } catch (IOException | IllegalStateException ex) {
                emitters.remove(emitter);   // client gone
            }
        }
    }
}
