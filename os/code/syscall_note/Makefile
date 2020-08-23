
hello: hello.s
	gcc -m32 -o hello hello.s

run-hello: hello
	@./hello; echo "exit code is $$?"


.PHONY: clean
clean:
		$(RM) hello
