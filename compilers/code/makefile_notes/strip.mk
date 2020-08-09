
a :=   hello    world   #
$(info 1$(a)2) # 1hello    world   2

s := $(strip $(a))
$(info 1$(s)2) # 1hello world2

all:
