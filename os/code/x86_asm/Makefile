
datatype.s: datatype.c
	$(CC) -S -fno-asynchronous-unwind-tables -o $@ $<

hello_world: hello_world.s
	as --32 -o hello_world.o hello_world.s
	ld -m elf_i386 --dynamic-linker /lib/ld-linux.so.2 -o hello_world hello_world.o -lc
	rm hello_world.o

hello_main: hello_main.s
	gcc -m32 -o hello_main hello_main.s

.PHONY: clean
clean:
	$(RM) hello_world
	$(RM) hello_main
