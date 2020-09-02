
all: a.out


lex.yy.c: detect.l
	flex $<

a.out: lex.yy.c
	$(CC) $<

clean:
	$(RM) lex.yy.c a.out
