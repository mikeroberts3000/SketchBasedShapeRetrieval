# DEBUG = y
TRIMESHDIR = ../trimesh2
MAKERULESDIR = $(TRIMESHDIR)
INCLUDES = -I$(TRIMESHDIR)/include
LIBDIR = -L$(TRIMESHDIR)/lib.$(UNAME)

include $(MAKERULESDIR)/Makerules
ifeq ($(UNAME),Win32)
  COPTS += -mwindows
endif

SOURCES = rtsc.cc apparentridge.cc
OFILES = $(addprefix $(OBJDIR)/,$(SOURCES:.cc=.o))
PROGS = rtsc$(EXE)

default: $(PROGS)

LIBS += -ltrimesh
rtsc$(EXE) : $(OBJDIR)/rtsc.o $(OBJDIR)/apparentridge.o
	$(LINK) $(GLLIBS)

clean :
	-rm -f $(OFILES) $(OBJDIR)/Makedepend $(OBJDIR)/*.d
	-rm -rf $(OBJDIR)/ii_files
	-rmdir $(OBJDIR)

spotless : clean
	-rm -f $(PROGS)
	-rmdir $(DESTDIR)

VERSION = 1.5

tar:
	cd .. && tar zcvf rtsc-$(VERSION).tar.gz `find rtsc-$(VERSION) -name 'OBJ*' -prune -o -name CVS -prune -o -type f -print | sort`

zip:
	cd .. && find rtsc-$(VERSION) -name 'OBJ*' -prune -o -name CVS -prune -o -print | sort | zip -9 rtsc-$(VERSION).zip -@

