
KALDI_ROOT := $(HOME)/open-source/kaldi
include $(KALDI_ROOT)/src/kaldi.mk


EXTRA_CXXFLAGS += -I$(KALDI_ROOT)/src
EXTRA_CXXFLAGS += -I$(KALDI_ROOT)/tools/portaudio/install/include
EXTRA_CXXFLAGS += -I..
EXTRA_CXXFLAGS += -O2

EXTRA_LDFLAGS += -L../lib
EXTRA_LDLIBS +=

CXXFLAGS := $(filter-out -g,$(CXXFLAGS))
LDFLAGS := $(filter-out -g,$(LDFLAGS))

OBJFILES :=
LIBNAME := test_kaldi_lib

TESTFILES := test_kaldi
TESTFILES += test_event_map
TESTFILES += test_tree

USE_LIBS := $(KALDI_ROOT)/src/lib/kaldi-base.so
USE_LIBS += $(KALDI_ROOT)/src/lib/kaldi-tree.so
USE_LIBS += $(KALDI_ROOT)/src/lib/kaldi-feat.so
USE_LIBS += $(KALDI_ROOT)/src/lib/kaldi-matrix.so
USE_LIBS += $(KALDI_ROOT)/src/lib/kaldi-transform.so
USE_LIBS += $(KALDI_ROOT)/src/lib/kaldi-util.so

ADDLIBS += $(USE_LIBS)


include $(KALDI_ROOT)/src/makefiles/default_rules.mk
