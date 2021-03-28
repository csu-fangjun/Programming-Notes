package main

import "fmt"

func main() {
	var s = "initial" // s is a variable. Its type is inferred from the assignment
	fmt.Println(s)    // initial

	// we can declare several variables in one line
	var s1, s2 = "h1", "h2"
	fmt.Println(s1, s2) // h1 h2

	// we can specify the type of the variables
	var i, k int = 10, 20
	fmt.Println(i, k) // 10 20

	// Note that we can define a variable but do not assign
	// a value to it.
	//
	// b is default initialized
	var b bool
	fmt.Println(b) // false

	// define a variable and give it an initial value
	// the type of the variable is inferred automatically
	//
	// equivalent to
	//  var a int = 100
	// or
	//  var a = 100
	a := 100
}
