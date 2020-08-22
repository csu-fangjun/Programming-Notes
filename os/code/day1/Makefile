

disk.img: boot.s sample.s
	as -o boot.o boot.s
	as -o sample.o sample.s
	ld -o boot.bin --oformat binary -Ttext 0x7c00 boot.o
	ld -o sample.bin --oformat binary -Ttext 0x500 sample.o
	dd if=/dev/zero of=disk.img bs=512 count=2880 # make a 1.44 MB disk
	dd conv=notrunc if=boot.bin of=disk.img bs=512 count=1 seek=0
	dd conv=notrunc if=sample.bin of=disk.img bs=512 count=1 seek=1
	chmod a=rwx disk.img

.PHONY: clean run
clean:
	$(RM) *.bin *.o *.img

run: disk.img
	qemu-system-i386 -display curses -machine q35 -fda disk.img -gdb tcp::1234  -S
	# qemu-system-x86_64 -display curses -monitor telnet:127.0.0.1:1234,server,nowait -drive format=raw,file=boot.bin
	# telnet 127.0.0.1 1234
