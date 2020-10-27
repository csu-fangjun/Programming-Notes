package main

import "fmt"

func main() {
	i := 1
	s := 0
	for i <= 3 {
		s += i
		i += 1
	}
	fmt.Println("s is", s) // s is 6

	// we can define the variable `j` inside the for loop
	//
	// Note there is only j++, no ++j
	s = 0
	for j := 1; j <= 3; j++ {
		s += j
	}
	fmt.Println("s is", s) // s is 6

	s = 0
	/*
	     for var j = 1; j <= 3; j++ {  // we cannot use var in for !
	       s += j
	     }
	   	fmt.Println("s is", s) // s is 6
	*/

	for { // infinite loop
		fmt.Println("for loop")
		break
	}

	s = 0
	for k := 0; k < 6; k++ {
		if k%2 == 0 {
			continue // we've seen break and continue!
		}
		s += k
	}
	fmt.Println("s is", s) // s is 9

}
