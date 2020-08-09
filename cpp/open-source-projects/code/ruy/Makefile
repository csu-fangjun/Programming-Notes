
.PHONY: all
all:
	bazel run //:hello_ruy

.PHONY: profile
profile:
	bazel run --define ruy_profiler=true //:hello_profiler -s
