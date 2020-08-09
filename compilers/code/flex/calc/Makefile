
all: calc

lex.yy.c: calc.l calc.tab.h
	flex calc.l

# use -d so that it will generate the header file calc.tab.h
# `bison calc.y` generates only calc.tab.c
calc.tab.c calc.tab.h: calc.y
	bison -d calc.y

calc: calc.tab.c calc.tab.h lex.yy.c
	gcc -o calc calc.tab.c lex.yy.c -ll

clean:
	$(RM) calc.tab.c calc.tab.h lex.yy.c calc

