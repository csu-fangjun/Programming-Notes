%{
#include <stdio.h>

int num_lines = 0;
int num_chars = 0;
%}

%%
[\n] {++num_lines;}
. {++num_chars;}

%%

int main() {
  yylex();
  printf("num of lines: %d\n", num_lines);
  printf("num of characters: %d\n", num_chars);
}
