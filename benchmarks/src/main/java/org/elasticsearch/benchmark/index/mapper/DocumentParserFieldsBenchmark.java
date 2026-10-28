/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.benchmark.index.mapper;

import org.elasticsearch.common.bytes.BytesArray;
import org.elasticsearch.common.logging.LogConfigurator;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.index.mapper.LuceneDocument;
import org.elasticsearch.index.mapper.MapperService;
import org.elasticsearch.index.mapper.SourceToParse;
import org.elasticsearch.xcontent.XContentType;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Measures the per-field overhead of {@code DocumentParser} for documents with many mapped leaf fields
 * under an object, either as scalar values or as small arrays, with stored or synthetic {@code _source}.
 */
@Fork(value = 1)
@Warmup(iterations = 5)
@Measurement(iterations = 5)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Benchmark)
public class DocumentParserFieldsBenchmark {

    static {
        LogConfigurator.configureESLogging();
    }

    @Param({ "50" })
    private int fieldCount;

    @Param({ "scalar", "array" })
    private String valueShape;

    @Param({ "stored", "synthetic" })
    private String sourceMode;

    private MapperService mapperService;
    private SourceToParse sourceToParse;

    @Setup
    public void setUp() {
        StringBuilder mapping = new StringBuilder("{\"_doc\":{\"properties\":{\"metrics\":{\"properties\":{");
        StringBuilder doc = new StringBuilder("{\"metrics\":{");
        for (int i = 0; i < fieldCount; i++) {
            if (i > 0) {
                mapping.append(',');
                doc.append(',');
            }
            mapping.append("\"field_").append(i).append("\":{\"type\":\"long\"}");
            doc.append("\"field_").append(i).append("\":");
            switch (valueShape) {
                case "scalar" -> doc.append(i);
                case "array" -> doc.append('[').append(i).append(',').append(i + 1).append(']');
                default -> throw new IllegalArgumentException("unknown value shape [" + valueShape + "]");
            }
        }
        mapping.append("}}}}}");
        doc.append("}}");

        this.mapperService = MapperServiceFactory.create(
            mapping.toString(),
            List.of(),
            Settings.builder().put("index.mapping.source.mode", sourceMode).build()
        );
        this.sourceToParse = new SourceToParse("1", new BytesArray(doc.toString()), XContentType.JSON);
    }

    @Benchmark
    public List<LuceneDocument> parse() {
        return mapperService.documentMapper().parse(sourceToParse).docs();
    }
}
