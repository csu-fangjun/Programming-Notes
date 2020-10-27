package main

import "fmt"

func main() {
	/*
	  Note that we cannot use
	    if 7%2 {}
	  Integer is not implicitly converted to bool in go!
	*/

	// parenthese are not needed for conditions!
	if 7%2 == 1 {
		fmt.Println("7 is even")
	} else {
		fmt.Println("7 is odd")
	}

	if true {
		fmt.Println("An if without an else")
	}

	// note that we can define a variable in if!
	if n := 9; n < 0 {
		fmt.Println(n, "is negative")
	} else if n < 10 {
		// n is still accessible!
		fmt.Println(n, "is less than 10")
	} else {
		fmt.Println(n, "is greater than 9")
	}
}
