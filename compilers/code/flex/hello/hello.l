
%%

"hello world" {printf("Good bye");}
"exit" {printf("exiting.\n"); return 0;}

. {printf("unknown ");}
%%

/* if we define yywrap(); then there is no need to link to `-ll`. */
/* int yywrap() {return 1;} */

/*Even if libfl.a defines `main()`, it is not an error to define our own main() */
/* int main() { */
/*   yylex(); */
/*   return 0; */
/* } */
