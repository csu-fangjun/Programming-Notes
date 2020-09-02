
%%

"hello world" {printf("Good bye");}
"exit" {printf("exiting.\n"); return 0;}

%%

/* if we define yywrap(); then there is no need to link to `-ll`. */
/* int yywrap() {return 1;} */

/*Even if libfl.a defines `main()`, it is not an error to define our own main() */
/* int main() { */
/*   while(yylex()) {} */
/*   return 0; */
/* } */
