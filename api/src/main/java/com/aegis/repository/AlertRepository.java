package com.aegis.repository;

import com.aegis.entity.Alert;
import java.util.List;
import java.util.Optional;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

public interface AlertRepository extends JpaRepository<Alert, Long> {
    List<Alert> findByCaseId(Long caseId);
    List<Alert> findByCaseIdIsNullAndDismissedFalse(Pageable pageable);
    Optional<Alert> findByDatasetIdAndNodeId(Long datasetId, int nodeId);
}
