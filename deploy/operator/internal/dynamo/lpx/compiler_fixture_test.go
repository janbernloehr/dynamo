/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"testing"

	"capnproto.org/go/capnp/v3"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/common/v1"
	deploymentcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/deployment/v1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/testfixture"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

const (
	otherFixtureValue  = "other"
	v2TestBuildName    = "v2-test-build"
	v3HXTopologyFamily = "HX16x8x2"
	v3OpaqueTopology   = "opaque-v3-topology"
)

type staticModelRegistry map[string]*Build

func (staticModelRegistry) BuildURL(id string) (*url.URL, error) {
	return nil, fmt.Errorf("unexpected build URL lookup for %q", id)
}
func (staticModelRegistry) EnsureDownloaded(_ context.Context, buildURL url.URL) (bool, error) {
	return false, fmt.Errorf("unexpected build download for %q", buildURL.String())
}
func (staticModelRegistry) EnsureCheckpointDownloaded(_ context.Context, checkpoint *Checkpoint) (bool, error) {
	return false, fmt.Errorf("unexpected checkpoint download for %q", CheckpointKey(checkpoint))
}

func (source staticModelRegistry) AcquireBuild(_ context.Context, buildID string) (*Build, error) {
	snapshot := source[buildID]
	if snapshot == nil {
		return nil, fmt.Errorf("unknown test build %q", buildID)
	}
	return snapshot, nil
}

func newSelectedTestDGD(t *testing.T, name string, components ...v1beta1.DynamoComponentDeploymentSharedSpec) *v1beta1.DynamoGraphDeployment {
	t.Helper()

	t.Log("Construct one fresh selected-LPX projection fixture around the caller-owned configuration")
	return &v1beta1.DynamoGraphDeployment{
		Name: name, Namespace: "test", UID: "uid", Generation: 1,
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: components,
		},
	}
}

// testLPXComponent constructs an independent component with the caller's role membership.
func testLPXComponent(name, buildID string, roles ...v1beta1.ComponentRoleSpec) v1beta1.DynamoComponentDeploymentSharedSpec {
	return v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: name,
		ComponentType: v1beta1.ComponentTypeLPX,
		LPX:           &v1beta1.LPXConfig{BuildID: buildID},
		Roles:         roles,
	}
}

// testLPXPodTemplate constructs an independent role template for each caller.
func testLPXPodTemplate(image string) *corev1.PodTemplateSpec {
	return &corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: commonconsts.MainContainerName, Image: image}}},
	}
}

func acquireTestSnapshot(t *testing.T, buildDir string) *Build {
	t.Helper()
	registry, err := NewModelRegistry("", nil)
	require.NoError(t, err)
	snapshot, err := registry.AcquireBuild(t.Context(), buildDir)
	require.NoError(t, err)
	return snapshot
}

func projectTestBuild(t *testing.T, snapshot *Build, pipeline Pipeline) *ModelProjection {
	t.Helper()

	t.Log("Project the caller-owned normalized build for the selected pipeline")
	projectionBatch, err := appendModelProjections(nil, ModelProjectionInput{
		Pipeline: pipeline, Models: []string{"default"}, Build: snapshot,
	})
	require.NoError(t, err)
	return projectionBatch[0]
}

func writeV2CompilerFixture(t *testing.T) string {
	t.Helper()
	return writeCompilerFixture(t, newV2CompilerFixture())
}

func newV2CompilerFixture() testV3CapnpFixture {
	fixture := newV3CompilerFixture()
	fixture.architecture = "polaris"
	fixture.buildDirectoryName = v2TestBuildName
	fixture.units = []testLPUUnit{{Reservations: []testLPUReservation{testNodesReservation(2, 7), testNodesReservation(2, 8)}, Connectors: []testLPUConnector{{}}}}
	fixture.partitions = []testV3CapnpPartition{
		{id: 7, deviceType: commoncapnp.DeviceType_lpu, topology: registryTestTopology, numChips: 16, devicesPerNode: 8},
		{id: 8, deviceType: commoncapnp.DeviceType_lpu, topology: registryTestTopology, numChips: 16, devicesPerNode: 8},
	}
	return fixture
}

func writeV3CompilerFixture(t *testing.T) string {
	t.Helper()
	return writeCompilerFixture(t, newV3CompilerFixture())
}

type testV3CapnpFixture struct {
	buildDirectoryName string
	contractRevision   uint32
	pipelineName       string
	compilationMode    deploymentcapnp.CompilationMode
	architecture       string
	units              []testLPUUnit
	partSelect         bool
	cpuEmbeddings      bool
	partitions         []testV3CapnpPartition
}

type testV3CapnpPartition struct {
	id             uint32
	deviceType     commoncapnp.DeviceType
	topology       string
	numChips       uint32
	devicesPerNode uint32
}

func newV3CompilerFixture() testV3CapnpFixture {
	return testV3CapnpFixture{
		contractRevision: deploymentcapnp.CurrentContractRevision,
		units:            testSingleUnits(testNodesReservation(1, 1)),
		pipelineName:     "default",
		compilationMode:  deploymentcapnp.CompilationMode_lpuOnly,
		architecture:     "polarisB0",
		partitions: []testV3CapnpPartition{{
			id:             1,
			deviceType:     commoncapnp.DeviceType_lpu,
			topology:       v3OpaqueTopology,
			numChips:       16,
			devicesPerNode: 16,
		}},
	}
}

func writeCompilerFixture(t *testing.T, fixture testV3CapnpFixture) string {
	t.Helper()

	t.Log("Create the selected compiler fixture directory")
	buildDir := t.TempDir()
	if fixture.buildDirectoryName != "" {
		buildDir = filepath.Join(buildDir, fixture.buildDirectoryName)
		require.NoError(t, os.Mkdir(buildDir, 0o700))
	}

	t.Log("Write the units deployment fixture")
	writeTestV3CapnpManifest(t, buildDir, fixture)
	return buildDir
}

func writeTestV3CapnpManifest(t *testing.T, buildDir string, fixture testV3CapnpFixture) {
	t.Helper()

	t.Log("Encode the deployment header")
	message, segment := capnp.NewSingleSegmentMessage(nil)
	manifest, err := deploymentcapnp.NewRootDeployment(segment)
	require.NoError(t, err)
	manifest.SetContractRevision(fixture.contractRevision)
	deployment, err := manifest.NewConfig()
	require.NoError(t, err)
	deployment.SetCompilationMode(fixture.compilationMode)
	program, err := deployment.NewProgram()
	require.NoError(t, err)
	require.NoError(t, program.SetProgramName(fixture.pipelineName))
	program.SetSupportsCpuEmbeddings(fixture.cpuEmbeddings)
	program.SetStandaloneTokenEmbeddings(fixture.cpuEmbeddings)
	runtimeIO, err := deployment.NewRuntimeIo()
	require.NoError(t, err)
	runtimeIO.SetProtocol(deploymentcapnp.RuntimeIoProtocol_fpgaRoce)
	runtimeIO.SetIoFpgaCount(1)
	runtimeIO.SetFanoutFactor(1)

	testfixture.WriteUnits(t, deployment, fixture.units)

	if fixture.cpuEmbeddings {
		assets, err := manifest.NewRuntimeAssets()
		require.NoError(t, err)
		require.NoError(t, assets.SetTokenEmbeddingsPath("part-0/text_embeddings.npz"))
	}
	if fixture.partSelect {
		partSelect, err := manifest.NewPartSelect()
		require.NoError(t, err)
		selected, err := partSelect.NewPartitions(1)
		require.NoError(t, err)
		selected.At(0).SetDeviceType(commoncapnp.DeviceType_lpu)
		selected.At(0).SetPartitionId(fixture.partitions[0].id)
	}
	partitions, err := manifest.NewPartitions(int32(len(fixture.partitions)))
	require.NoError(t, err)
	for index, fixturePartition := range fixture.partitions {
		partition := partitions.At(index)
		ref, err := partition.NewPartition()
		require.NoError(t, err)
		ref.SetDeviceType(fixturePartition.deviceType)
		ref.SetPartitionId(fixturePartition.id)
		switch fixturePartition.deviceType {
		case commoncapnp.DeviceType_lpu:
			detail, err := partition.Detail().NewLpu()
			require.NoError(t, err)
			require.NoError(t, detail.SetPath(fmt.Sprintf("part-%d", fixturePartition.id)))
			require.NoError(t, detail.SetTopology(fixturePartition.topology))
			detail.SetNumChips(fixturePartition.numChips)
			detail.SetDevicesPerNode(fixturePartition.devicesPerNode)
			architecture := deploymentcapnp.LpuArchitecture_lp20
			if fixture.architecture == "polarisB0" {
				architecture = deploymentcapnp.LpuArchitecture_lp30
			}
			detail.SetArchitecture(architecture)

		case commoncapnp.DeviceType_cuda:
			_, err := partition.Detail().NewCuda()
			require.NoError(t, err)
		case commoncapnp.DeviceType_cpu:
			_, err := partition.Detail().NewCpu()
			require.NoError(t, err)
		default:
			t.Fatalf("unsupported test device type %q", fixturePartition.deviceType)
		}
	}

	t.Log("Publish the units deployment")
	data, err := message.Marshal()
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(buildDir, gbuildDeploymentV1CapnpFile), data, 0o600))
}
