package com.aegis.entity;

import jakarta.persistence.*;

/** A registered dataset/model the app can analyse (spec §8.5). */
@Entity
@Table(name = "dataset")
public class Dataset {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, unique = true)
    private String name;
    private String variant;
    private long numNodes;
    private long numEdges;
    private long numIllicit;

    protected Dataset() {}

    public Dataset(String name, String variant, long numNodes, long numEdges, long numIllicit) {
        this.name = name; this.variant = variant;
        this.numNodes = numNodes; this.numEdges = numEdges; this.numIllicit = numIllicit;
    }

    public double illicitRatio() { return numNodes == 0 ? 0 : (double) numIllicit / numNodes; }

    public Long getId() { return id; }
    public String getName() { return name; }
    public String getVariant() { return variant; }
    public long getNumNodes() { return numNodes; }
    public long getNumEdges() { return numEdges; }
    public long getNumIllicit() { return numIllicit; }
}
