package main

import (
	"log"
	"os"
	"path/filepath"
)

func main() {
	exe1, err := os.Executable()
	if err != nil {
		log.Fatalf("os.Executable failed: %s", err)
	}
	println("os.Executable()=" + exe1)

	exe2, err := filepath.Abs(exe1)
	if err != nil {
		log.Fatalf("filepath.Abs failed: %s", err)
	}
	println("filepath.Abs()=" + exe2)

	dir1 := filepath.Join(exe2, "../..")
	println("dir1=" + dir1)
	dir2 := filepath.Join(exe2, "../../..")
	println("dir2=" + dir2)
}
