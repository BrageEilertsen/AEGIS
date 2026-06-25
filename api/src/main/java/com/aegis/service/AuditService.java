package com.aegis.service;

import com.aegis.entity.AuditHead;
import com.aegis.entity.AuditRecord;
import com.aegis.repository.AuditHeadRepository;
import com.aegis.repository.AuditRecordRepository;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.HexFormat;
import java.util.List;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/** Tamper-evident audit trail (regulated-domain requirement / EU AI Act traceability). Every entry
 *  is hashed together with the previous entry's hash, forming a chain: altering, reordering, or
 *  deleting any past record changes its hash and breaks every later link, which {@link #verify()}
 *  detects and points to. Appends serialize on a locked head row so the chain can't fork. */
@Service
public class AuditService {

    static final String GENESIS = "0".repeat(64);

    private final AuditRecordRepository records;
    private final AuditHeadRepository heads;

    public AuditService(AuditRecordRepository records, AuditHeadRepository heads) {
        this.records = records; this.heads = heads;
    }

    @Transactional
    public AuditRecord append(String actor, String action, String subjectType, Long subjectId, String payload) {
        AuditHead head = heads.lockHead().orElseGet(() -> heads.save(new AuditHead(1L, 0, GENESIS)));
        long seq = head.getLastSeq() + 1;
        Instant at = Instant.now().truncatedTo(ChronoUnit.MILLIS);
        String prev = head.getLastHash();
        String hash = hash(seq, at.toEpochMilli(), actor, action, subjectType, subjectId, payload, prev);

        AuditRecord rec = records.save(
                new AuditRecord(seq, at, actor, action, subjectType, subjectId, payload, prev, hash));
        head.setLastSeq(seq);
        head.setLastHash(hash);
        heads.save(head);
        return rec;
    }

    /** Recompute the whole chain and report the first break (if any). */
    @Transactional(readOnly = true)
    public Verification verify() {
        String prev = GENESIS;
        long expectedSeq = 1;
        long count = 0;
        for (AuditRecord r : records.findAllByOrderBySeqAsc()) {
            count++;
            if (r.getSeq() != expectedSeq) {
                return Verification.broken(r.getSeq(), "sequence gap (expected " + expectedSeq + ")");
            }
            if (!prev.equals(r.getPrevHash())) {
                return Verification.broken(r.getSeq(), "chain break: prevHash does not match");
            }
            String recomputed = hash(r.getSeq(), r.getAtMillis(), r.getActor(), r.getAction(),
                    r.getSubjectType(), r.getSubjectId(), r.getPayload(), r.getPrevHash());
            if (!recomputed.equals(r.getHash())) {
                return Verification.broken(r.getSeq(), "hash mismatch: record was altered");
            }
            prev = r.getHash();
            expectedSeq++;
        }
        return Verification.ok(count, prev);
    }

    @Transactional(readOnly = true)
    public List<AuditRecord> forSubject(String subjectType, Long subjectId) {
        return records.findBySubjectTypeAndSubjectIdOrderBySeqAsc(subjectType, subjectId);
    }

    // --- hashing ---
    private static String hash(long seq, long atMillis, String actor, String action, String subjectType,
                               Long subjectId, String payload, String prevHash) {
        // Length-prefixed fields so no field's content can be confused with a delimiter.
        StringBuilder sb = new StringBuilder();
        field(sb, Long.toString(seq));
        field(sb, Long.toString(atMillis));
        field(sb, actor);
        field(sb, action);
        field(sb, subjectType);
        field(sb, subjectId == null ? "" : subjectId.toString());
        field(sb, payload == null ? "" : payload);
        field(sb, prevHash);
        return sha256(sb.toString());
    }

    private static void field(StringBuilder sb, String v) {
        sb.append(v.length()).append(':').append(v).append('|');
    }

    private static String sha256(String s) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            return HexFormat.of().formatHex(md.digest(s.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 unavailable", e);   // never on a standard JVM
        }
    }

    /** Result of a chain verification. */
    public record Verification(boolean valid, long records, String headHash,
                               Long brokenAtSeq, String reason) {
        static Verification ok(long count, String head) {
            return new Verification(true, count, head, null, null);
        }
        static Verification broken(long seq, String reason) {
            return new Verification(false, seq - 1, null, seq, reason);
        }
    }
}
