/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"hash"
)

const (
	workloadSetDigestVersion = "dynamo-lpx-compilation-set/v1"
	maxLPXPartitions         = 256
)

// appendModelProjections appends one component's immutable model projections to
// the caller-owned destination, which may be nil. Existing elements are unchanged.
// intent.Build contains a normalized, non-nil build; Models is nonempty
// and Pipeline is selected by the validated resolver. Source inputs are not mutated;
// discard error results.
func appendModelProjections(dst []*ModelProjection, intent ModelProjectionInput) ([]*ModelProjection, error) {
	projections, err := appendUnitModelProjections(dst, intent)
	if err != nil {
		return nil, err
	}

	// Only a hybrid build's Cyborg conductor consumes a checkpoint.
	var checkpoint *Checkpoint
	if intent.Pipeline == PipelineHybrid {
		if err := intent.Build.checkpointErr; err != nil {
			return nil, err
		}
		checkpoint = intent.Build.checkpoint
	}
	for _, projection := range projections[len(dst):] {
		projection.checkpoint = checkpoint
	}

	// Bound the component's physical build before runtime expansion and request publication.
	projection := projections[len(dst)]
	boundedPartitions := len(projection.partitions)
	if len(projection.localPartitionIDs) > 0 {
		// A local selection may leave no remote partitions, but the build itself stays bounded.
		boundedPartitions = len(intent.Build.partitions)
	} else if boundedPartitions < 1 {
		return nil, fmt.Errorf("LPX projection has %d partitions, limit is 1..%d", boundedPartitions, maxLPXPartitions)
	}
	if boundedPartitions > maxLPXPartitions {
		return nil, fmt.Errorf("LPX projection has %d partitions, limit is 1..%d", boundedPartitions, maxLPXPartitions)
	}
	return projections, nil
}

func workloadSetDigest(projections []*ModelProjection) (WorkloadDigest, error) {
	first := projections[0]
	if len(projections) == 1 {
		return first.digest, nil
	}
	transcript := &digestTranscript{hash: sha256.New()}
	transcript.field("schema", []byte(workloadSetDigestVersion))
	for _, projection := range projections {
		if projection.configuredBuild.family != first.configuredBuild.family {
			return WorkloadDigest{}, fmt.Errorf(
				"LPX model projections have mixed target families %q and %q",
				first.configuredBuild.family,
				projection.configuredBuild.family,
			)
		}
		transcript.field("model", []byte(projection.model))
		// Keep the versioned field tag stable so this terminology refactor does not
		// introduce an additional aggregate identity change beyond the base PR.
		transcript.field("compilation-digest", projection.digest[:])
	}
	return transcript.sum(), nil
}

type digestTranscript struct {
	hash hash.Hash
}

func (t *digestTranscript) field(tag string, value []byte) {
	writeLengthDelimited(t.hash, []byte(tag))
	writeLengthDelimited(t.hash, value)
}

func (t *digestTranscript) sum() WorkloadDigest {
	var digest WorkloadDigest
	copy(digest[:], t.hash.Sum(nil))
	return digest
}

func writeLengthDelimited(writer hash.Hash, value []byte) {
	var length [8]byte
	binary.BigEndian.PutUint64(length[:], uint64(len(value)))
	_, _ = writer.Write(length[:])
	_, _ = writer.Write(value)
}
