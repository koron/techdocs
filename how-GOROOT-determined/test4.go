package main

import (
	"fmt"
	"log"
	"syscall"
	"unicode/utf16"
)

func getLongPathName(name string) (string, error) {
	in := utf16.Encode([]rune(name))
	in = append(in, 0)
	out := make([]uint16, syscall.MAX_PATH)
	n, err := syscall.GetLongPathName(&in[0], &out[0], uint32(len(out)))
	if err != nil {
		return "", err
	}
	return string(utf16.Decode(out[:n])), nil
}

func main() {
	var err error
	s := `D:\go\current\`
	s, err = getLongPathName(s)
	if err != nil {
		log.Fatalf("failed: %s", err)
	}
	fmt.Printf("s=%s\n", s)
}
