
.PHONY: all clean

all: hello

# the default output of `flex` is lex.yy.c
lex.yy.c: hello.l
	# flex hello.l
	lex hello.l

hello: lex.yy.c
	# gcc -o hello lex.yy.c -lfl
	gcc -o hello lex.yy.c  -ll

clean:
	$(RM) hello lex.yy.c
