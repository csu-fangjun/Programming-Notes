package main

import (
	"fmt"
)

const s string = "constant"

func main() {
	fmt.Println(s) // constant

	// Note that `const` replaces `var`
	const n = 10 // its type is inferred
	const d = 1e2 / n
	fmt.Println(n, d)     // 10 10
	fmt.Println(int64(n)) // 10
}
