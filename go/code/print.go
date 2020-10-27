package main

import "fmt"

func main() {
	fmt.Print("hello")    // no new line
	fmt.Println(" world") // a new line
	fmt.Println("hello" + " world")
	fmt.Println("hello", "world") // note hello and world are separated by a space!
	// the above code print: hello world

	fmt.Println(true)          // true
	fmt.Println(false)         // false
	fmt.Println(!false)        // true
	fmt.Println(false || true) // true
	fmt.Println(false && true) // false

	fmt.Println("3/2 = ", 3/2) // 3/2 = 1. Note it is 1
}
