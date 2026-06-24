package com.aegis.controller;

import com.aegis.dto.DatasetDto;
import com.aegis.service.DatasetService;
import java.util.List;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/datasets")
public class DatasetController {

    private final DatasetService service;

    public DatasetController(DatasetService service) { this.service = service; }

    @GetMapping
    public List<DatasetDto> list() { return service.list(); }

    @GetMapping("/{id}")
    public DatasetDto get(@PathVariable Long id) { return service.get(id); }
}
