
PKG_CONFIG_PATH := $(PKG_CONFIG_PATH):$(HOME)/software/pcre/lib/pkgconfig

# -I/path/to/pcre/include
CXXFLAGS := $(shell pkg-config libpcre --cflags)
CXXFLAGS += -std=c++11

CXXFLAGS += $(shell pkg-config libpcre --cflags)

# -L/path/to/pcre/lib -lpcre
LDFLAGS := $(shell pkg-config libpcre --libs)

# -L/path/to/pcre/lib -lpcre -lpcrecpp
LDFLAGS += $(shell pkg-config libpcrecpp --libs)

example: example.cc

.PHONY: clean
clean:
	$(RM) example
