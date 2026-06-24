package com.aegis.repository;

import com.aegis.entity.CachedExplanation;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

public interface CachedExplanationRepository extends JpaRepository<CachedExplanation, Long> {
    Optional<CachedExplanation> findByDatasetIdAndNodeId(Long datasetId, int nodeId);
}
