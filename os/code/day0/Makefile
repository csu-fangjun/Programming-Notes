
boot.bin: boot.s
	as -o boot.o boot.s
	objcopy -O binary boot.o boot.bin

.PHONY: clean run
clean:
	$(RM) *.bin *.o

run: boot.bin
	qemu-system-x86_64 -display curses -monitor telnet:127.0.0.1:1234,server,nowait -drive format=raw,file=boot.bin
	# telnet 127.0.0.1 1234
