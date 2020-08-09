%{
  int words = 0;
  int chars = 0;
  int lines = 0;
%}

%%
[^ \t\r\n\f\v]+ {++words; chars += strlen(yytext); printf("detected: %s, %ld\n", yytext, yyleng);}
\n {++lines; ++chars; }
. {++chars;}
%%

int main() {
  yylex();
  printf("words: %d, chars: %d, lines: %d\n", words, chars, lines);
  return 0;
}
