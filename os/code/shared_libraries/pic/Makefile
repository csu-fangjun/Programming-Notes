
libtest.so: test.cc
	g++ -fPIC -c test.cc
	g++ -shared -o libtest.so test.o

main.o: main.cc
	g++ -c -o main.o main.cc

main: main.o libtest.so
	g++ -o main main.o -L$(PWD) -l test

.PHONY: clean
clean:
	$(RM) main *.so *.o *.s
