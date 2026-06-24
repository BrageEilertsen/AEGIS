package com.aegis.repository;

import com.aegis.entity.Dataset;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

public interface DatasetRepository extends JpaRepository<Dataset, Long> {
    Optional<Dataset> findByName(String name);
}
