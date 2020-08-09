%{
#include <stdio.h>
#include <unistd.h> // for getlogin()
%}
%%
username {printf("%s", getlogin());}
%%
// by default, it prints everything that does not match.
