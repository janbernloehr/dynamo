/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"capnproto.org/go/capnp/v3"
	commoncapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/common/v1"
	deploymentcapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/deployment/v1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/testfixture"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func newDeploymentV1ContractFixture(t *testing.T) deploymentcapnpv1.Deployment {
	t.Helper()
	_, seg := capnp.NewSingleSegmentMessage(nil)
	deployment, err := deploymentcapnpv1.NewRootDeployment(seg)
	require.NoError(t, err)
	deployment.SetContractRevision(deploymentcapnpv1.CurrentContractRevision)
	config, err := deployment.NewConfig()
	require.NoError(t, err)
	config.SetCompilationMode(deploymentcapnpv1.CompilationMode_lpx)
	_, err = config.NewProgram()
	require.NoError(t, err)
	runtimeIO, err := config.NewRuntimeIo()
	require.NoError(t, err)
	runtimeIO.SetProtocol(deploymentcapnpv1.RuntimeIoProtocol_fpgaRoce)
	runtimeIO.SetIoFpgaCount(4)
	runtimeIO.SetFanoutFactor(2)
	partitions, err := deployment.NewPartitions(1)
	require.NoError(t, err)
	partition, err := partitions.At(0).NewPartition()
	require.NoError(t, err)
	partition.SetDeviceType(commoncapnpv1.DeviceType_lpu)
	detail, err := partitions.At(0).Detail().NewLpu()
	require.NoError(t, err)
	require.NoError(t, detail.SetPath("part-0"))
	require.NoError(t, detail.SetTopology(registryTestTopology))
	detail.SetArchitecture(deploymentcapnpv1.LpuArchitecture_lp20)
	detail.SetNumChips(8)
	detail.SetDevicesPerNode(8)
	testfixture.WriteUnits(t, config, testSingleUnits(testNodesReservation(1, 0)))
	return deployment
}

// testUnitBundle describes a deployment.v1 contract with compiler-owned LPU units.
type testUnitBundle struct {
	architecture   deploymentcapnpv1.LpuArchitecture
	devicesPerNode uint32
	ioFPGACount    uint32
	ioFanoutFactor uint32
	// hybrid compiles in lpx mode and adds one CUDA partition after the LPU partitions.
	hybrid  bool
	lpxMode bool
	// cpuEmbeddings sets supportsCpuEmbeddings and standaloneTokenEmbeddings.
	cpuEmbeddings bool
	partitions    []testUnitPartition
	// partSelect lists the selected LPU partitions; nil writes a full build.
	partSelect []uint32
	// units is written as config.lpuUnits; nil leaves the field unset.
	units []testLPUUnit
}

type testUnitPartition struct {
	id       uint32
	numChips uint32
	// topology names the compiled topology; empty uses registryTestTopology.
	topology string
}

type testLPUUnit = testfixture.Unit
type testLPUReservation = testfixture.Reservation
type testLPUConnector = testfixture.Connector

// testUnitPartitions returns LPU partitions that share one geometry.
func testUnitPartitions(numChips uint32, ids ...uint32) []testUnitPartition {
	partitions := make([]testUnitPartition, len(ids))
	for index, id := range ids {
		partitions[index] = testUnitPartition{id: id, numChips: numChips}
	}
	return partitions
}

func testNodesReservation(nodes uint32, ids ...uint32) testLPUReservation {
	return testLPUReservation{IDs: ids, Nodes: nodes}
}

func testHXReservation(traysPerHemisphere, hemispheres, racks uint8, ids ...uint32) testLPUReservation {
	return testLPUReservation{IDs: ids, HX: []uint8{traysPerHemisphere, hemispheres, racks}}
}

// testSingleUnits places each reservation in its own unit.
func testSingleUnits(reservations ...testLPUReservation) []testLPUUnit {
	units := make([]testLPUUnit, len(reservations))
	for index, reservation := range reservations {
		units[index] = testLPUUnit{Reservations: []testLPUReservation{reservation}}
	}
	return units
}

// testHXConventionConnector joins a source reservation's last host, device by
// device, to the destination's first host, with lanes [4, 2, 1].
func testHXConventionConnector(sourceHosts, devicesPerNode uint32) testLPUConnector {
	connector := testLPUConnector{HX: true}
	for device := range devicesPerNode {
		connector.Pairs = append(connector.Pairs, [2]uint32{(sourceHosts-1)*devicesPerNode + device, device})
	}
	return connector
}

// encodeTestUnitBundle serializes the bundle as deployment.v1.capnp.bin bytes.
func encodeTestUnitBundle(t *testing.T, bundle testUnitBundle) []byte {
	t.Helper()
	message, segment := capnp.NewSingleSegmentMessage(nil)
	deployment, err := deploymentcapnpv1.NewRootDeployment(segment)
	require.NoError(t, err)
	deployment.SetContractRevision(deploymentcapnpv1.CurrentContractRevision)

	t.Log("Encode the program and runtime I/O")
	config, err := deployment.NewConfig()
	require.NoError(t, err)
	config.SetCompilationMode(deploymentcapnpv1.CompilationMode_lpuOnly)
	if bundle.hybrid || bundle.lpxMode {
		config.SetCompilationMode(deploymentcapnpv1.CompilationMode_lpx)
	}
	program, err := config.NewProgram()
	require.NoError(t, err)
	program.SetSupportsCpuEmbeddings(bundle.cpuEmbeddings)
	program.SetStandaloneTokenEmbeddings(bundle.cpuEmbeddings)
	runtimeIO, err := config.NewRuntimeIo()
	require.NoError(t, err)
	runtimeIO.SetIoFpgaCount(max(1, bundle.ioFPGACount))
	runtimeIO.SetFanoutFactor(max(1, bundle.ioFanoutFactor))

	t.Log("Encode the compiler-owned units")
	if bundle.units != nil {
		testfixture.WriteUnits(t, config, bundle.units)
	}

	t.Log("Encode the partition selection, the LPU partitions and the hybrid CUDA partition")
	if bundle.partSelect != nil {
		selection, err := deployment.NewPartSelect()
		require.NoError(t, err)
		selected, err := selection.NewPartitions(int32(len(bundle.partSelect)))
		require.NoError(t, err)
		for index, id := range bundle.partSelect {
			selected.At(index).SetDeviceType(commoncapnpv1.DeviceType_lpu)
			selected.At(index).SetPartitionId(id)
		}
	}
	count := len(bundle.partitions)
	if bundle.hybrid {
		count++
	}
	partitions, err := deployment.NewPartitions(int32(count))
	require.NoError(t, err)
	lastID := uint32(0)
	for index, partition := range bundle.partitions {
		ref, err := partitions.At(index).NewPartition()
		require.NoError(t, err)
		ref.SetDeviceType(commoncapnpv1.DeviceType_lpu)
		ref.SetPartitionId(partition.id)
		lastID = max(lastID, partition.id)
		detail, err := partitions.At(index).Detail().NewLpu()
		require.NoError(t, err)
		require.NoError(t, detail.SetPath(fmt.Sprintf("part-%d", partition.id)))
		topology := partition.topology
		if topology == "" {
			topology = registryTestTopology
		}
		require.NoError(t, detail.SetTopology(topology))
		detail.SetArchitecture(bundle.architecture)
		if len(bundle.units) == 0 {
			detail.SetNumChips(partition.numChips)
		}
		detail.SetDevicesPerNode(bundle.devicesPerNode)

	}
	if bundle.hybrid {
		ref, err := partitions.At(len(bundle.partitions)).NewPartition()
		require.NoError(t, err)
		ref.SetDeviceType(commoncapnpv1.DeviceType_cuda)
		ref.SetPartitionId(lastID + 1)
		_, err = partitions.At(len(bundle.partitions)).Detail().NewCuda()
		require.NoError(t, err)
	}

	data, err := message.Marshal()
	require.NoError(t, err)
	return data
}

// acquireTestUnitBundle publishes the bundle in a fresh build directory and
// acquires it through the model registry.
func acquireTestUnitBundle(t *testing.T, bundle testUnitBundle) (*Build, error) {
	t.Helper()
	root := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(root, gbuildDeploymentV1CapnpFile), encodeTestUnitBundle(t, bundle), 0o600))
	registry, err := NewModelRegistry("", nil)
	require.NoError(t, err)
	return registry.AcquireBuild(t.Context(), root)
}

func TestDeploymentContractIgnoresRuntimeProtocol(t *testing.T) {
	t.Log("Normalize a deployment with FPGA runtime I/O")
	deployment := newDeploymentV1ContractFixture(t)
	want, err := buildFromGbuildDeploymentV1("gs://models/build", deployment)
	require.NoError(t, err)
	config, err := deployment.Config()
	require.NoError(t, err)
	runtimeIO, err := config.RuntimeIo()
	require.NoError(t, err)

	for _, test := range []struct {
		name     string
		protocol deploymentcapnpv1.RuntimeIoProtocol
	}{
		{"host", deploymentcapnpv1.RuntimeIoProtocol_host},
		{"unknown", deploymentcapnpv1.RuntimeIoProtocol(99)},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Preserve the normalized build when only the runtime protocol changes")
			runtimeIO.SetProtocol(test.protocol)
			got, err := buildFromGbuildDeploymentV1("gs://models/build", deployment)
			require.NoError(t, err)
			require.Equal(t, want, got)
		})
	}
}

func TestDeploymentContractRejectsInvalidFacts(t *testing.T) {
	for _, name := range []string{"revision", "future revision", "zero endpoints", "zero fanout", "unknown device", "missing architecture", "unknown architecture"} {
		t.Run(name, func(t *testing.T) {
			t.Log("Alter one typed deployment fact and retain the other valid fields")
			deployment := newDeploymentV1ContractFixture(t)
			config, err := deployment.Config()
			require.NoError(t, err)
			runtimeIO, err := config.RuntimeIo()
			require.NoError(t, err)
			var wantErr string
			switch name {
			case "missing architecture", "unknown architecture":
				partitions, err := deployment.Partitions()
				require.NoError(t, err)
				detail, err := partitions.At(0).Detail().Lpu()
				require.NoError(t, err)
				architecture := deploymentcapnpv1.LpuArchitecture_unspecified
				if name == "unknown architecture" {
					architecture = deploymentcapnpv1.LpuArchitecture(99)
				}
				detail.SetArchitecture(architecture)
				wantErr = "unsupported chip architecture"
			case "revision":
				deployment.SetContractRevision(0)
				wantErr = "contractRevision"
			case "future revision":
				deployment.SetContractRevision(99)
				wantErr = "contractRevision"
			case "zero endpoints":
				runtimeIO.SetIoFpgaCount(0)
				wantErr = "ioFpgaCount"
			case "zero fanout":
				runtimeIO.SetFanoutFactor(0)
				wantErr = "fanoutFactor"
			case "unknown device":
				partitions, err := deployment.Partitions()
				require.NoError(t, err)
				ref, err := partitions.At(0).Partition()
				require.NoError(t, err)
				ref.SetDeviceType(commoncapnpv1.DeviceType(99))
				wantErr = "deviceType"
			}

			t.Log("Reject the malformed scheduler input before producing a build")
			_, err = buildFromGbuildDeploymentV1("gs://models/build", deployment)
			require.ErrorContains(t, err, wantErr)
		})
	}
}

func TestDeploymentContractAcquisitionAndRuntimePath(t *testing.T) {
	for _, test := range []struct {
		name         string
		architecture deploymentcapnpv1.LpuArchitecture
		family       BuildFamily
		topology     string
		chips        uint32
	}{
		{"LP20", deploymentcapnpv1.LpuArchitecture_lp20, BuildFamilyXT, registryTestTopology, 8},
		{"LP30", deploymentcapnpv1.LpuArchitecture_lp30, BuildFamilyHX, "HX16x8x2", 16},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Materialize deployment and opaque execution without a report or JSON")
			root := t.TempDir()
			deployment := newDeploymentV1ContractFixture(t)
			partitions, err := deployment.Partitions()
			require.NoError(t, err)
			detail, err := partitions.At(0).Detail().Lpu()
			require.NoError(t, err)
			detail.SetArchitecture(test.architecture)
			require.NoError(t, detail.SetTopology(test.topology))
			detail.SetNumChips(test.chips)
			detail.SetDevicesPerNode(test.chips)
			payload, err := deployment.Message().Marshal()
			require.NoError(t, err)
			require.NoError(t, os.WriteFile(filepath.Join(root, gbuildDeploymentV1CapnpFile), payload, 0o600))
			require.NoError(t, os.WriteFile(filepath.Join(root, "execution.v1.capnp.bin"), []byte("opaque runtime bytes"), 0o600))
			registry, err := NewModelRegistry("", nil)
			require.NoError(t, err)

			t.Log("Acquire and normalize exactly the scheduler contract")
			snapshot, err := registry.AcquireBuild(t.Context(), root)
			require.NoError(t, err)
			normalized := snapshot
			require.NoError(t, err)
			require.EqualValues(t, 4, normalized.ioFPGACount)
			require.Equal(t, test.family, normalized.family)

			t.Log("Pass the selected deployment file to Cyborg before authored environment references")
			projections, err := appendModelProjections(nil, ModelProjectionInput{
				Pipeline: PipelineHybrid, Models: []string{"default"}, Build: normalized,
			})
			require.NoError(t, err)
			require.Len(t, projections, 1)
			projection := projections[0]
			require.Equal(t, test.family, projection.configuredBuild.family)
			container := corev1.Container{Env: []corev1.EnvVar{{Name: "MODEL_PATH", Value: "$(GBUILD_MANIFEST_PATH)"}}}
			require.NoError(t, applyCyborgManifestPath(&container, projection, "/models"))
			require.Equal(t, []corev1.EnvVar{
				{Name: gbuildManifestPathEnv, Value: filepath.Join(root, gbuildDeploymentV1CapnpFile)},
				{Name: "MODEL_PATH", Value: "$(GBUILD_MANIFEST_PATH)"},
			}, container.Env)

			t.Log("Bind exact deployment bytes to snapshot identity")
			identity, err := deployment.NewIdentity()
			require.NoError(t, err)
			require.NoError(t, identity.SetBuildId("different-build"))
			changed, err := deployment.Message().Marshal()
			require.NoError(t, err)
			require.NoError(t, os.WriteFile(filepath.Join(root, gbuildDeploymentV1CapnpFile), changed, 0o600))
			second, err := registry.AcquireBuild(t.Context(), root)
			require.NoError(t, err)
			require.NotEqual(t, snapshot.contentID, second.contentID)
		})
	}
}

func TestDeploymentContractRejectsDirectory(t *testing.T) {
	t.Parallel()
	t.Log("Publish a directory at the required deployment path")
	root := t.TempDir()
	require.NoError(t, os.Mkdir(filepath.Join(root, gbuildDeploymentV1CapnpFile), 0o700))
	registry, err := NewModelRegistry("", nil)
	require.NoError(t, err)

	t.Log("Reject the directory before decoding a deployment")
	_, err = registry.AcquireBuild(t.Context(), root)
	require.ErrorContains(t, err, "not a regular file")
}

func TestDeploymentContractReadsRemoteBytesOnce(t *testing.T) {
	t.Log("Serve one deployment contract with a separate opaque execution artifact")
	payload, err := newDeploymentV1ContractFixture(t).Message().Marshal()
	require.NoError(t, err)
	client := &fakeModelServiceClient{
		files: map[string]*fakeModelFileStream{
			gbuildDeploymentV1CapnpFile: {chunks: modelFileChunks(gbuildDeploymentV1CapnpFile, string(payload))},
		},
	}
	registry, err := NewModelRegistry("gs://bucket/registry", client)
	require.NoError(t, err)
	_, err = registry.AcquireBuild(t.Context(), "model/build")
	require.NoError(t, err)
	require.Empty(t, client.listRequests)
	require.Len(t, client.filesRequests, 1)
	require.Equal(t, []string{gbuildDeploymentV1CapnpFile}, client.filesRequests[0].GetFileSelector().GetPaths())
}

func TestDeploymentContractBoundsAcquisition(t *testing.T) {
	t.Log("Create a sparse deployment file beyond the existing metadata budget")
	root := t.TempDir()
	file, err := os.Create(filepath.Join(root, gbuildDeploymentV1CapnpFile))
	require.NoError(t, err)
	require.NoError(t, file.Truncate(int64(maxBuildSnapshotMetadataBytes)+1))
	require.NoError(t, file.Close())
	registry, err := NewModelRegistry("", nil)
	require.NoError(t, err)

	t.Log("Reject oversized metadata before a decoder or runtime sees it")
	_, err = registry.AcquireBuild(t.Context(), root)
	require.ErrorIs(t, err, errBuildFileTooLarge)
}

func TestDeploymentContractRejectsCorruptNewOnly(t *testing.T) {
	t.Log("Acquire stable corrupt deployment bytes with no legacy entry point")
	root := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(root, gbuildDeploymentV1CapnpFile), []byte("corrupt"), 0o600))
	registry, err := NewModelRegistry("", nil)
	require.NoError(t, err)
	_, err = registry.AcquireBuild(t.Context(), root)
	require.ErrorContains(t, err, "parsing deployment.v1.capnp.bin")
}

func TestDeploymentRejectsRevisionOneBeforeReadingMovedFields(t *testing.T) {
	t.Parallel()
	t.Log("Reject old-layout bytes before reading the new deployment layout")
	// These are the original revision-1 bytes of the subnode8 geometry fixture.
	data, err := os.ReadFile("testdata/deployment-revision-1.capnp.bin")
	require.NoError(t, err)
	root := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(root, gbuildDeploymentV1CapnpFile), data, 0o600))
	registry, err := NewModelRegistry("", nil)
	require.NoError(t, err)
	_, err = registry.AcquireBuild(t.Context(), root)
	require.ErrorContains(t, err, "contractRevision = 1, required 2; recompile the model")
}
