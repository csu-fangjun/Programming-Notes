

main: main.o main.lds
	ld -m elf_i386 -o main -T main.lds main.o

main.o: main.c
	#$(CC) -c -m32 -g main.c
	$(CC) -c -m32 main.c
	# objdump -S -d main.o

clean:
	$(RM) main.o main

