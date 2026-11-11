/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestGBuildGeometryContract(t *testing.T) {
	t.Log("Read the offline GBuild corpus and its geometry index")
	root := filepath.Join("testdata", "gbuild_geometry_contract")
	data, err := os.ReadFile(filepath.Join(root, "fixtures.json"))
	require.NoError(t, err)
	var index struct {
		Fixtures []struct {
			Name       string `json:"name"`
			SHA256     string `json:"sha256"`
			Partitions []struct {
				ID             int      `json:"id"`
				Topology       string   `json:"topology"`
				NumChips       int      `json:"num_chips"`
				DevicesPerNode int      `json:"devices_per_node"`
				SourceExtent   []uint32 `json:"source_extent"`
			} `json:"partitions"`
			UnitPartitionIDs []int `json:"unit_partition_ids"`
		} `json:"fixtures"`
	}
	require.NoError(t, json.Unmarshal(data, &index))

	t.Log("Keep the scheduling and runtime expectations independent of the projector")
	want := map[string]struct {
		ids       []int
		chips     []int
		extents   [][]int64
		agents    int
		contentID string
	}{
		"subnode8": {
			[]int{7}, []int{8}, [][]int64{{16, 1, 1, 1}}, 1,
			"sha256:9d9d004e82b3d433d91f9b51323c64976735d06e2ac55532b07790be9913d65f",
		},
		"cross_hemi32": {
			[]int{7}, []int{32}, [][]int64{{16, 1, 2, 1}}, 2,
			"sha256:f4a3ebcaec8ab17a9ece4e3e44563ae6489119822c9cfb38a17696093f91e328",
		},
		"rack256": {
			[]int{7}, []int{256}, [][]int64{{16, 8, 2, 1}}, 16,
			"sha256:dc11b3e1c3982a494988c1ba10d8b5d0080a3867e8a40808aaa2f24697c42b7a",
		},
		"connected_chain": {
			[]int{7, 8}, []int{32, 16}, [][]int64{{16, 1, 2, 1}, {16, 1, 1, 1}}, 3,
			"sha256:eaa1fddb7feb724b64b0a9adb301a780dd8e559680c2ff31d40aa9127b7c7c23",
		},
	}
	require.Len(t, index.Fixtures, len(want))
	seen := make(map[string]bool)
	for _, fixture := range index.Fixtures {
		t.Run(fixture.Name, func(t *testing.T) {
			t.Log("Verify the frozen bytes and their compiler-owned facts before projection")
			expected, ok := want[fixture.Name]
			require.True(t, ok)
			require.False(t, seen[fixture.Name], "duplicate corpus entry")
			seen[fixture.Name] = true
			fixtureRoot, err := filepath.Abs(filepath.Join(root, fixture.Name))
			require.NoError(t, err)
			payload, err := os.ReadFile(filepath.Join(fixtureRoot, gbuildDeploymentV1CapnpFile))
			require.NoError(t, err)
			require.LessOrEqual(t, len(payload), 64*1024)
			require.Equal(t, fixture.SHA256, fmt.Sprintf("%x", sha256.Sum256(payload)))
			deployment, err := decodeGbuildDeploymentV1(payload)
			require.NoError(t, err)
			require.EqualValues(t, 2, deployment.ContractRevision())
			raw, err := deployment.Partitions()
			require.NoError(t, err)
			require.Equal(t, len(expected.ids), raw.Len())
			require.Len(t, fixture.Partitions, len(expected.ids))
			for i, source := range fixture.Partitions {
				ref, err := raw.At(i).Partition()
				require.NoError(t, err)
				require.EqualValues(t, expected.ids[i], ref.PartitionId())
				require.Equal(t, expected.ids[i], source.ID)
				require.Equal(t, expected.chips[i], source.NumChips)
				require.Equal(t, 16, source.DevicesPerNode)
				detail, err := raw.At(i).Detail().Lpu()
				require.NoError(t, err)
				topology, err := detail.Topology()
				require.NoError(t, err)
				require.Equal(t, v3OpaqueTopology, topology)
				require.Equal(t, v3OpaqueTopology, source.Topology)
				require.EqualValues(t, expected.chips[i], detail.NumChips())
				require.EqualValues(t, 16, detail.DevicesPerNode())
			}

			t.Log("Acquire the fixture's exact source identity and project the LPU-only workload")
			build := acquireTestSnapshot(t, fixtureRoot)
			require.Equal(t, expected.contentID, build.contentID)
			require.Equal(t, build.contentID, acquireTestSnapshot(t, fixtureRoot).contentID)
			projections, err := appendModelProjections(nil, ModelProjectionInput{
				Pipeline: PipelineSingle, Models: []string{"default"}, Build: build,
			})
			require.NoError(t, err)
			require.Len(t, projections, 1)
			projection := projections[0]
			require.Equal(t, expected.agents, projection.AgentReplicas())
			require.Equal(t, expected.contentID, projection.CompilerSnapshotDigest())
			spec := projection.RequestSpec(&MaterializationPlan{}, "agents")
			require.Len(t, spec.Partitions, len(expected.ids))
			require.Len(t, spec.NodeLocal.PartitionMappings, len(expected.ids))
			for i, partition := range spec.Partitions {
				require.EqualValues(t, i, partition.Ordinal)
				require.Equal(t, fmt.Sprintf("partition-%03d", i), partition.ID)
				require.NotNil(t, partition.Extent)
				require.Equal(t, expected.extents[i], *partition.Extent)
				require.EqualValues(t, i, spec.NodeLocal.PartitionMappings[i].ModelPartitionID)
				require.Equal(t, partition.ID, spec.NodeLocal.PartitionMappings[i].PartitionID)
				require.Equal(t, expected.ids[i], build.partitions[i].sourcePartitionID)
				require.Equal(t, 16, build.partitions[i].devicesPerNode)
			}

			t.Log("Retain the producer's ordered unit members")
			config, err := deployment.Config()
			require.NoError(t, err)
			runtimeIO, err := config.RuntimeIo()
			require.NoError(t, err)
			require.EqualValues(t, 1, runtimeIO.IoFpgaCount())
			require.EqualValues(t, 1, runtimeIO.FanoutFactor())
			units, err := config.LpuUnits()
			require.NoError(t, err)
			require.Equal(t, 1, units.Len())
			reservations, err := units.At(0).Reservations()
			require.NoError(t, err)
			require.Equal(t, len(expected.ids), reservations.Len())
			for i := range reservations.Len() {
				ids, err := reservations.At(i).PartitionIds()
				require.NoError(t, err)
				require.Equal(t, 1, ids.Len())
				require.EqualValues(t, expected.ids[i], ids.At(0))
				hx, err := reservations.At(i).Size().Hx()
				require.NoError(t, err)
				require.Equal(t, expected.extents[i][1:], []int64{int64(hx.TraysPerHemisphere()), int64(hx.Hemispheres()), int64(hx.Racks())})
			}
			require.Equal(t, expected.ids, fixture.UnitPartitionIDs)

			t.Log("Honor the compiler-selected chain at the final logical-node boundary")
			if fixture.Name != "connected_chain" {
				require.Empty(t, spec.PropSyncConnectors)
				return
			}
			require.Equal(t, []int{7, 8}, fixture.UnitPartitionIDs)
			require.Len(t, spec.PropSyncConnectors, 1)
			connector := spec.PropSyncConnectors[0]
			require.Equal(t, "partition-000", connector.FromPartitionID)
			require.Equal(t, "partition-001", connector.ToPartitionID)
			require.NotNil(t, connector.Requirement.AcceptableLaneMultiplicities)
			require.Equal(t, []int64{4, 2, 1}, *connector.Requirement.AcceptableLaneMultiplicities)
			require.NotNil(t, connector.Requirement.Connections)
			require.Len(t, *connector.Requirement.Connections, 16)
			for i, connection := range *connector.Requirement.Connections {
				require.EqualValues(t, 16+i, connection.FromLogicalDevice)
				require.EqualValues(t, i, connection.ToLogicalDevice)
			}
		})
	}
}
