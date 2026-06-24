package com.aegis.service;

import com.aegis.dto.DatasetDto;
import com.aegis.entity.Dataset;
import com.aegis.exception.NotFoundException;
import com.aegis.repository.DatasetRepository;
import java.util.List;
import org.springframework.stereotype.Service;

@Service
public class DatasetService {

    private final DatasetRepository repo;

    public DatasetService(DatasetRepository repo) { this.repo = repo; }

    public List<DatasetDto> list() { return repo.findAll().stream().map(this::toDto).toList(); }

    public DatasetDto get(Long id) {
        return repo.findById(id).map(this::toDto)
                .orElseThrow(() -> new NotFoundException("dataset " + id + " not found"));
    }

    private DatasetDto toDto(Dataset d) {
        return new DatasetDto(d.getId(), d.getName(), d.getVariant(), d.getNumNodes(),
                d.getNumEdges(), d.getNumIllicit(), d.illicitRatio());
    }
}
