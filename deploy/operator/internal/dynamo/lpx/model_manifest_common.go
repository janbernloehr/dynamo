/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"path/filepath"
	"strings"
)

func cleanManifestRelativeBuildPath(field, rawPath string) (string, error) {
	if strings.ContainsAny(rawPath, "\x00\r\n") {
		return "", fmt.Errorf("%s %q must not contain NUL bytes or line breaks", field, rawPath)
	}
	assetPath := strings.TrimSpace(rawPath)
	if assetPath == "" {
		return "", fmt.Errorf("%s is empty", field)
	}
	if filepath.IsAbs(assetPath) {
		return "", fmt.Errorf("%s %q must be relative and stay within build directory", field, rawPath)
	}
	assetPath = filepath.ToSlash(filepath.Clean(assetPath))
	if assetPath == "." || assetPath == ".." || strings.HasPrefix(assetPath, "../") {
		return "", fmt.Errorf("%s %q must be relative and stay within build directory", field, rawPath)
	}
	return assetPath, nil
}

func positiveManifestUInt32ToInt(field string, value uint32) (int, error) {
	if value == 0 {
		return 0, fmt.Errorf("%s must be >= 1, got 0", field)
	}
	return int(value), nil
}
