.PHONY: clean
%: %.c
	gcc -fsanitize=address -O1 -fno-omit-frame-pointer -g $< -o $@

%: %.cc
	gcc -fsanitize=address -O1 -fno-omit-frame-pointer -g $< -o $@

clean:
	$(RM) ex{1,2,3,4,5,6,7,8}
