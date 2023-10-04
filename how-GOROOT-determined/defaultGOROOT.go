package main

import (
	"bufio"
	"debug/pe"
	"flag"
	"fmt"
	"io"
	"log"
	"strings"
)

func extractDefaultGOROOT(name string, fn func(*pe.File, int, *pe.Symbol) error) error {
	f, err := pe.Open(name)
	if err != nil {
		return err
	}
	defer f.Close()
	for i, sym := range f.Symbols {
		if !strings.Contains(sym.Name, "defaultGOROOT") {
			continue
		}
		err := fn(f, i, sym)
		if err != nil {
			return err
		}
	}
	return nil
}

func main() {
	flag.Parse()
	if flag.NArg() != 1 {
		log.Fatal("need exact an argument for executable file")
	}
	name := flag.Arg(0)
	err := extractDefaultGOROOT(name, func(f *pe.File, i int, sym *pe.Symbol) error {
		fmt.Printf("#%d %+v\n", i, sym)

		r := f.Sections[sym.SectionNumber].Open()
		_, err := r.Seek(int64(sym.Value), io.SeekStart)
		if err != nil {
			return fmt.Errorf("failed to seek %d: %w", sym.Value, err)
		}
		b, err := bufio.NewReader(r).ReadSlice(0)
		fmt.Printf("  %d bytes read from section #%d", len(b), sym.SectionNumber)
		if err != nil {
			return err
		}
		s := string(b[:len(b)-1]) // truncate trailing '\0'
		fmt.Printf("  %s\n", s)
		// TODO:
		return nil
	})
	if err != nil {
		log.Fatalf("failed to extraction %s: %s", name, err)
	}
}
