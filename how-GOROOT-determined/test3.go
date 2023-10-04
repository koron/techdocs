package main

import "strings"

func main() {
	a := "D:\\Go\\current"
	b := "D:\\go\\current"
	println("strings.EqualFold():", strings.EqualFold(a, b))
}
