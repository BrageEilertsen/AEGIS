package com.aegis.dto;

/** Public view of a registered dataset. */
public record DatasetDto(Long id, String name, String variant, long numNodes, long numEdges,
                         long numIllicit, double illicitRatio) {}
