package com.aegis.repository;

import com.aegis.entity.CaseActivity;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;

public interface CaseActivityRepository extends JpaRepository<CaseActivity, Long> {
    List<CaseActivity> findByCaseIdOrderByAtAsc(Long caseId);
}
