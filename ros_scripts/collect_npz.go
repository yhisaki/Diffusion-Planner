// collect_npz.go
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

func collectNPZUnderNamedDirs(root string, dirName string) ([]string, error) {
	var paths []string

	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}

		if !d.IsDir() || d.Name() != dirName {
			return nil
		}

		err = filepath.WalkDir(path, func(npzPath string, npzEntry os.DirEntry, err error) error {
			if err != nil {
				return err
			}

			if npzEntry.IsDir() {
				return nil
			}

			if filepath.Ext(npzPath) == ".npz" {
				absPath, err := filepath.Abs(npzPath)
				if err != nil {
					return err
				}
				paths = append(paths, absPath)
			}

			return nil
		})

		return err
	})

	sort.Strings(paths)
	return paths, err
}

func writeJSON(path string, data []string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	encoder := json.NewEncoder(f)
	encoder.SetIndent("", "  ")
	return encoder.Encode(data)
}

func main() {
	saveDirFlag := flag.String("save_dir", "", "Directory to save path_list_train.json and path_list_valid.json")
	flag.Parse()

	if flag.NArg() != 1 {
		fmt.Fprintf(os.Stderr, "Usage: %s DATASET_ROOT [--save_dir SAVE_DIR]\n", os.Args[0])
		os.Exit(1)
	}

	datasetRoot, err := filepath.Abs(flag.Arg(0))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	saveDir := datasetRoot
	if *saveDirFlag != "" {
		saveDir, err = filepath.Abs(*saveDirFlag)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}

	if err := os.MkdirAll(saveDir, 0755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	trainPaths, err := collectNPZUnderNamedDirs(datasetRoot, "train")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	validPaths, err := collectNPZUnderNamedDirs(datasetRoot, "valid")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	trainJSONPath := filepath.Join(saveDir, "path_list_train.json")
	validJSONPath := filepath.Join(saveDir, "path_list_valid.json")

	if err := writeJSON(trainJSONPath, trainPaths); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	if err := writeJSON(validJSONPath, validPaths); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	fmt.Printf("Train npz files: %d\n", len(trainPaths))
	fmt.Printf("Valid npz files: %d\n", len(validPaths))
	fmt.Printf("Saved: %s\n", trainJSONPath)
	fmt.Printf("Saved: %s\n", validJSONPath)
}
