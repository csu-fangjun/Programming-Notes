%{
#include <stdlib.h> // for atoi, atof

extern void match_id(); // defined in the following

// detect patterns for the C language

%}
identifier  [_a-zA-Z][a-zA-Z0-9_]*
digit [0-9]
%%

{digit}+ { printf("an integer: %d\n", atoi(yytext)); }

{digit}+"."{digit}*  { printf("a float: %f\n", atof(yytext));}

if|else|for|while {printf("a keyword: %s\n", yytext);}

exit {printf("exiting!\n"); return 0;}

  /* note that we can call a function inside the function! */
{identifier} {printf("an ID: %s\n", yytext); match_id();}

"+"|"-"|"*"|"/" {printf("an operator: %s\n", yytext);}

; {printf("semicolon: %s\n", yytext);}

  /* c++ stle comment. Note that | means the action for this rule is the same as the action of the next rule */
"/""/".*  |

  /* c style comment */
"/""*"([^*]|([*][^/]))*"*""/" {printf("a comment: %s\n", yytext);}

  /* not that we can use either \" or ["] to represent a double quote */
\"[^"]*["] {printf("a string: %s\n", yytext);}

\'.\' {printf("A character: %s\n", yytext);}

[ \t\n]+
. {printf("unknown character: %s\n", yytext);}

%%

// define yywrap() ourselves so that we do not need to link with `-lfl`
int yywrap() {return 1;}

void match_id() {
  printf("matched an id!\n");
}

int main(int argc, char* argv[]) {
  // if `yyin` is NULL, it will set to stdin inside `yylex`.
  if (argc > 1) {
    yyin = fopen(argv[1], "r");
  }
  yylex();
  return 0;
}
