
%token foo
%token bar

%union{
  int hello,
  double* world
}

%%

hello: foo

