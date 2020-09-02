
srcs := $(shell find . -type f -name "*.c" -printf "%f ")
targets :=$(srcs:%.c=%)

all: $(targets)

%: %.c
	$(CC) $(CFLAGS) -o $@ $<

run-%: %
	./$<

.PHONY: clean
clean:
	$(RM) $(targets)
	$(RM) a.out

