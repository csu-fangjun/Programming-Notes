

disk.img: boot.s hello.c
	as -o boot.o boot.s
	gcc -c -ffreestanding -nostdlib -m16 -o hello.o -gdwarf-4 -ggdb3 hello.c
	ld -o boot.bin --oformat binary -Ttext 0x7c00 boot.o
	ld -m elf_i386 -nmagic -o hello.elf -T hello.lds  hello.o
	objcopy -O binary hello.elf hello.bin
	# cp hello.elf hello.bin
	dd if=/dev/zero of=disk.img bs=512 count=2880 # make a 1.44 MB disk
	dd conv=notrunc if=boot.bin of=disk.img bs=512 count=1 seek=0
	dd conv=notrunc if=hello.bin of=disk.img bs=512 count=2 seek=1
	chmod a=rwx disk.img

.PHONY: clean run
clean:
	$(RM) *.bin *.o *.img *.elf

# note that we have to start gdb with
#   gdb hello.elf
run: disk.img
	qemu-system-i386 -display curses -machine q35 -fda disk.img -gdb tcp::1234  -S
	# qemu-system-x86_64 -display curses -monitor telnet:127.0.0.1:1234,server,nowait -drive format=raw,file=boot.bin
	# telnet 127.0.0.1 1234
