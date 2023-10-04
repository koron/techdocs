package main

import (
	"go/build"
	"os"
	"runtime"
)

func main() {
	println("os.Getenv(GOROOT)=" + os.Getenv("GOROOT"))
	println("runtime.GOROOT=" + runtime.GOROOT())
	println("build.Default.GOROOT=" + build.Default.GOROOT)
}
