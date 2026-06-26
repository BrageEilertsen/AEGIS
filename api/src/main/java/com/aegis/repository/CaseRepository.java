package com.aegis.repository;

import com.aegis.domain.CaseState;
import com.aegis.entity.CaseFile;
import java.util.List;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

public interface CaseRepository extends JpaRepository<CaseFile, Long> {
    List<CaseFile> findByState(CaseState state, Pageable pageable);
    List<CaseFile> findByAssignee(String assignee, Pageable pageable);
    List<CaseFile> findByStateAndAssignee(CaseState state, String assignee, Pageable pageable);
}
