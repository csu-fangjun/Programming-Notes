
all: ex.ll

%.ll: %.c
	clang -emit-llvm -S $< -o $@


clean:
	$(RM) *.ll *.bc *.s
