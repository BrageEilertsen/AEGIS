package com.aegis.repository;

import com.aegis.entity.AuditHead;
import jakarta.persistence.LockModeType;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Lock;
import org.springframework.data.jpa.repository.Query;

public interface AuditHeadRepository extends JpaRepository<AuditHead, Long> {

    /** Row-locks the single head row so concurrent appends serialize at the database. */
    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("select h from AuditHead h where h.id = 1")
    Optional<AuditHead> lockHead();
}
