/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FilterMergePolicy;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.MergePolicy;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.IORunnable;
import org.elasticsearch.common.logging.LogConfigurator;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.index.codec.vectors.diskbbq.es94.ES940DiskBBQVectorsFormat;
import org.elasticsearch.index.codec.vectors.diskbbq.es95.ES950DiskBBQVectorsFormat;
import org.elasticsearch.index.codec.vectors.diskbbq.next.ESNextDiskBBQVectorsFormat;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.apache.lucene.tests.index.BaseKnnVectorsFormatTestCase.randomVector;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.nullValue;

/**
 * Verifies that a <em>running</em> DiskBBQ (IVF) merge honours merge aborts, so that shard close, relocation and node
 * shutdown are not blocked until a potentially hours-long vector merge completes.
 * <p>
 * The abort is signalled the same way {@code IndexEngine} signals relocation aborts: by overriding
 * {@link MergePolicy.OneMerge#isAborted()}. Lucene's {@code MergeRateLimiter} reads the merge progress directly and never
 * sees such an override, so the only way the merge can stop is through explicit {@link MergeState#checkAborted()} calls
 * in the IVF writer. The abort is injected when the writer enters a given phase, detected by the temporary file it
 * creates for that phase.
 */
public class DiskBBQMergeAbortTests extends ESTestCase {

    static {
        LogConfigurator.configureESLogging();
    }

    private static final String FIELD = "vector";
    private static final int DIMENSIONS = 32;
    private static final int NUM_SEGMENTS = 3;
    private static final int DOCS_PER_SEGMENT = 300;
    private static final List<String> IVF_TEMP_FILE_SUFFIXES = List.of("ivfvec_", "ivfdoc_", "civf_", "qvec_");

    /** IVF merge phases, identified by the temporary file the writer creates when it enters them. */
    private enum Phase {
        VECTOR_DUMP("ivfvec_"),
        CLUSTERING("civf_"),
        POSTING_LISTS("qvec_");

        private final String tempFileSuffix;

        Phase(String tempFileSuffix) {
            this.tempFileSuffix = tempFileSuffix;
        }
    }

    private static List<KnnVectorsFormat> diskBBQFormats() {
        return List.of(
            new ES920DiskBBQVectorsFormat(64, 4),
            new ES940DiskBBQVectorsFormat(64, 4),
            new ES950DiskBBQVectorsFormat(64, 4),
            new ESNextDiskBBQVectorsFormat(64, 4, null)
        );
    }

    /**
     * Given a force merge of several DiskBBQ segments,
     * when the merge is aborted while the merged vectors are being dumped to a temporary file,
     * then the IVF merge stops with a {@link MergePolicy.MergeAbortedException} and leaves no temporary files behind.
     */
    public void testAbortWhileDumpingVectorsStopsMerge() throws IOException {
        assertAbortStopsMerge(Phase.VECTOR_DUMP);
    }

    /**
     * Given a force merge of several DiskBBQ segments,
     * when the merge is aborted while centroids are being computed,
     * then the IVF merge stops before writing the centroid index and posting lists.
     */
    public void testAbortWhileClusteringStopsMerge() throws IOException {
        assertAbortStopsMerge(Phase.CLUSTERING);
    }

    /**
     * Given a force merge of several DiskBBQ segments,
     * when the merge is aborted while posting lists are being built,
     * then the IVF merge stops before all posting lists are written.
     */
    public void testAbortWhileBuildingPostingListsStopsMerge() throws IOException {
        assertAbortStopsMerge(Phase.POSTING_LISTS);
    }

    /**
     * Given a force merge of several DiskBBQ segments,
     * when the merge is not aborted,
     * then the merge completes into a single segment holding every vector.
     */
    public void testMergeWithoutAbortCompletes() throws IOException {
        for (KnnVectorsFormat format : diskBBQFormats()) {
            // given
            try (MergeHarness harness = new MergeHarness(format, null)) {
                // when
                harness.forceMerge();

                // then
                assertThat(format.getName(), harness.recordingFormat.mergeFailure, nullValue());
                assertThat(format.getName(), harness.recordingFormat.mergeCompleted.get(), equalTo(true));
                try (DirectoryReader reader = DirectoryReader.open(harness.directory)) {
                    assertThat(format.getName(), reader.leaves().size(), equalTo(1));
                    assertThat(format.getName(), countVectors(reader), equalTo(NUM_SEGMENTS * DOCS_PER_SEGMENT));
                }
            }
        }
    }

    private void assertAbortStopsMerge(Phase phase) throws IOException {
        for (KnnVectorsFormat format : diskBBQFormats()) {
            String context = format.getName() + " aborted during " + phase;
            // given
            try (MergeHarness harness = new MergeHarness(format, phase)) {
                // when
                IOException e = expectThrows(
                    IOException.class,
                    context + ": force merge completed after the merge was aborted",
                    harness::forceMerge
                );

                // then
                assertThat(context + ": abort was never injected", harness.abortRequested.get(), equalTo(true));
                assertThat(context, e.getCause(), instanceOf(MergePolicy.MergeAbortedException.class));
                assertThat(
                    context + ": IVF mergeOneField ran to completion after the merge was aborted",
                    harness.recordingFormat.mergeFailure,
                    instanceOf(MergePolicy.MergeAbortedException.class)
                );
                assertThat(context + ": temp files leaked", harness.recordingFormat.liveTempFilesOnFailure, empty());
                try (DirectoryReader reader = DirectoryReader.open(harness.directory)) {
                    assertThat(context, reader.leaves().size(), greaterThan(1));
                    assertThat(context, countVectors(reader), equalTo(NUM_SEGMENTS * DOCS_PER_SEGMENT));
                }
            }
        }
    }

    private static int countVectors(DirectoryReader reader) throws IOException {
        int count = 0;
        for (LeafReaderContext leaf : reader.leaves()) {
            count += leaf.reader().getFloatVectorValues(FIELD).size();
        }
        return count;
    }

    /**
     * Builds an index of {@link #NUM_SEGMENTS} segments and force merges them, aborting the merge when the IVF writer
     * creates the temporary file of the given phase.
     */
    private class MergeHarness implements AutoCloseable {
        private final AtomicBoolean abortRequested = new AtomicBoolean();
        private final TempFileTrackingDirectory directory;
        private final RecordingKnnVectorsFormat recordingFormat;
        private final IndexWriter writer;

        MergeHarness(KnnVectorsFormat format, @Nullable Phase abortAt) throws IOException {
            directory = new TempFileTrackingDirectory(newDirectory(), abortAt == null ? null : abortAt.tempFileSuffix, abortRequested);
            recordingFormat = new RecordingKnnVectorsFormat(format, directory);
            IndexWriterConfig config = new IndexWriterConfig().setCodec(TestUtil.alwaysKnnVectorsFormat(recordingFormat))
                .setMergePolicy(new SingleForcedMergePolicy(abortRequested))
                .setMergeScheduler(new SerialMergeScheduler())
                .setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH)
                .setRAMBufferSizeMB(256);
            writer = new IndexWriter(directory, config);
            for (int s = 0; s < NUM_SEGMENTS; s++) {
                for (int d = 0; d < DOCS_PER_SEGMENT; d++) {
                    Document document = new Document();
                    document.add(new KnnFloatVectorField(FIELD, randomVector(DIMENSIONS), VectorSimilarityFunction.EUCLIDEAN));
                    writer.addDocument(document);
                }
                writer.commit();
            }
        }

        void forceMerge() throws IOException {
            writer.forceMerge(1);
            writer.commit();
        }

        @Override
        public void close() throws IOException {
            writer.close();
            directory.close();
        }
    }

    /**
     * Merge policy that never merges in the background and merges all segments into one on force merge, using a
     * {@link MergePolicy.OneMerge} whose {@link MergePolicy.OneMerge#isAborted()} is overridden, mirroring how
     * {@code IndexEngine} reports relocation aborts.
     */
    private static class SingleForcedMergePolicy extends FilterMergePolicy {
        private final AtomicBoolean abortRequested;

        SingleForcedMergePolicy(AtomicBoolean abortRequested) {
            super(NoMergePolicy.INSTANCE);
            this.abortRequested = abortRequested;
        }

        @Override
        public MergeSpecification findForcedMerges(
            SegmentInfos segmentInfos,
            int maxSegmentCount,
            Map<SegmentCommitInfo, Boolean> segmentsToMerge,
            MergeContext mergeContext
        ) {
            if (segmentInfos.size() <= maxSegmentCount) {
                return null;
            }
            MergeSpecification spec = new MergeSpecification();
            spec.add(new OneMerge(segmentInfos.asList()) {
                @Override
                public boolean isAborted() {
                    return abortRequested.get() || super.isAborted();
                }
            });
            return spec;
        }
    }

    /**
     * Requests the abort when the IVF writer creates the temporary file of the phase under test, and tracks IVF temporary
     * files that are still present so the test can assert they are cleaned up when the merge aborts.
     */
    private static class TempFileTrackingDirectory extends FilterDirectory {
        private final String abortOnTempFileSuffix;
        private final AtomicBoolean abortRequested;
        private final Set<String> liveTempFiles = ConcurrentHashMap.newKeySet();

        TempFileTrackingDirectory(Directory in, @Nullable String abortOnTempFileSuffix, AtomicBoolean abortRequested) {
            super(in);
            this.abortOnTempFileSuffix = abortOnTempFileSuffix;
            this.abortRequested = abortRequested;
        }

        @Override
        public IndexOutput createTempOutput(String prefix, String suffix, IOContext context) throws IOException {
            IndexOutput output = super.createTempOutput(prefix, suffix, context);
            if (IVF_TEMP_FILE_SUFFIXES.contains(suffix)) {
                liveTempFiles.add(output.getName());
            }
            if (suffix.equals(abortOnTempFileSuffix)) {
                abortRequested.set(true);
            }
            return output;
        }

        @Override
        public void deleteFile(String name) throws IOException {
            super.deleteFile(name);
            liveTempFiles.remove(name);
        }

        Set<String> liveTempFiles() {
            return Set.copyOf(liveTempFiles);
        }
    }

    /**
     * Wraps a DiskBBQ format to record how the IVF {@code mergeOneField} ends, so the test can tell that the abort was
     * honoured by the IVF writer itself rather than by a later abort check elsewhere in the merge, and can inspect the
     * temporary files left at that point. Reading is delegated to the wrapped format, which shares its name.
     */
    private static class RecordingKnnVectorsFormat extends KnnVectorsFormat {
        private final KnnVectorsFormat delegate;
        private final TempFileTrackingDirectory directory;
        private final AtomicBoolean mergeCompleted = new AtomicBoolean();
        private volatile Throwable mergeFailure;
        private volatile Set<String> liveTempFilesOnFailure = Set.of();

        RecordingKnnVectorsFormat(KnnVectorsFormat delegate, TempFileTrackingDirectory directory) {
            super(delegate.getName());
            this.delegate = delegate;
            this.directory = directory;
        }

        @Override
        public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
            KnnVectorsWriter writer = delegate.fieldsWriter(state);
            return new KnnVectorsWriter() {
                @Override
                public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
                    return writer.addField(fieldInfo);
                }

                @Override
                public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
                    writer.flush(maxDoc, sortMap);
                }

                @Override
                public IORunnable mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
                    try {
                        IORunnable deferred = writer.mergeOneField(fieldInfo, mergeState);
                        mergeCompleted.set(true);
                        return deferred;
                    } catch (Throwable t) {
                        mergeFailure = t;
                        liveTempFilesOnFailure = directory.liveTempFiles();
                        throw t;
                    }
                }

                @Override
                public void finish() throws IOException {
                    writer.finish();
                }

                @Override
                public void close() throws IOException {
                    writer.close();
                }

                @Override
                public long ramBytesUsed() {
                    return writer.ramBytesUsed();
                }
            };
        }

        @Override
        public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
            return delegate.fieldsReader(state);
        }

        @Override
        public int getMaxDimensions(String fieldName) {
            return delegate.getMaxDimensions(fieldName);
        }
    }
}
