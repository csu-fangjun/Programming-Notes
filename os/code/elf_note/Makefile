
all: hello

hello.o: hello.s
	gcc -m32 -c -o hello.o hello.s

hello: hello.o
	gcc -m32 -o hello hello.o

init.s: init.cc
	g++ -fno-asynchronous-unwind-tables -S -fno-dwarf2-cfi-asm -o init.s-bak init.cc
	cat init.s-bak | c++filt > init.s
	rm init.s-bak

run-hello: hello
	@./hello; echo "exit code is $$?"

.PHONY: clean
clean:
	$(RM) a.out hello.o hello
