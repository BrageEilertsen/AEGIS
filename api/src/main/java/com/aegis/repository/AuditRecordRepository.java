package com.aegis.repository;

import com.aegis.entity.AuditRecord;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;

public interface AuditRecordRepository extends JpaRepository<AuditRecord, Long> {
    List<AuditRecord> findAllByOrderBySeqAsc();
    List<AuditRecord> findBySubjectTypeAndSubjectIdOrderBySeqAsc(String subjectType, Long subjectId);
}
