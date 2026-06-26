package com.aegis.stream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.contains;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.atLeastOnce;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

import com.aegis.service.CaseService;
import java.time.Instant;
import org.junit.jupiter.api.Test;

/** The windowed pre-screen fires on a fan-out motif (and persists an alert) but stays quiet on
 *  ordinary one-off transfers. */
class StreamProcessorTest {

    private final StreamProperties props = new StreamProperties(true, 90, 700, 0.75, 60);
    private final StreamBroadcaster broadcaster = mock(StreamBroadcaster.class);
    private final CaseService cases = mock(CaseService.class);
    private final StreamProcessor processor = new StreamProcessor(props, broadcaster, cases);

    @Test
    void fanOutRaisesAlert() {
        Instant now = Instant.now();
        for (int i = 0; i < 8; i++) {   // one mule -> many distinct beneficiaries, fast
            processor.process(new TransactionEvent("t" + i, now, "MULE", "DST" + i, 500, "USD"));
        }
        verify(broadcaster, atLeastOnce()).publish(eq("alert"), any());
        verify(cases, atLeastOnce()).createAlert(eq(0L), anyInt(), anyDouble(), contains("fan-out"));
        assertThat(processor.stats().totalAlerts()).isGreaterThan(0);
    }

    @Test
    void ordinaryTrafficStaysQuiet() {
        Instant now = Instant.now();
        for (int i = 0; i < 6; i++) {   // distinct one-off transfers, no structure
            processor.process(new TransactionEvent("t" + i, now, "SRC" + i, "DST" + i, 400, "USD"));
        }
        verify(broadcaster, never()).publish(eq("alert"), any());
        assertThat(processor.stats().totalTransactions()).isEqualTo(6);
    }
}
