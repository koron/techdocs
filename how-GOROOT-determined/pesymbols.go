package main

import (
	"debug/pe"
	"flag"
	"fmt"
	"log"
)

func extractSymbols(name string) ([]*pe.Symbol, error) {
	f, err := pe.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return f.Symbols, nil
}

func main() {
	flag.Parse()
	if flag.NArg() != 1 {
		log.Fatal("need exact an argument for executable file")
	}
	name := flag.Arg(0)
	syms, err := extractSymbols(name)
	if err != nil {
		log.Fatalf("failed to extract symbols from %s: %s", name, err)
	}
	for i, sym := range syms {
		fmt.Printf("#%d %+v\n", i, sym)
	}
}
