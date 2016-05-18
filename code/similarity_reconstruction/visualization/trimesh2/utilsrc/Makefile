# DEBUG = y
MAKERULESDIR = ..
DESTDIR = ../bin.$(UNAME)
INCLUDES = -I../include
LIBDIR = -L../lib.$(UNAME)

include $(MAKERULESDIR)/Makerules

VIEWSOURCES =	mesh_view_modified.cc
VIEWOFILES = $(addprefix $(OBJDIR)/,$(VIEWSOURCES:.cc=.o))
VIEWPROG = $(DESTDIR)/mesh_view$(EXE)
PROGS = $(VIEWPROG)

default: $(PROGS)

LIBS += -ltrimesh

$(VIEWPROG) : $(VIEWOFILES)
	$(LINK) $(GLLIBS)

$(PROGS) : ../lib.$(UNAME)/libtrimesh.a

clean :
	-rm -f $(VIEWOFILES) $(OBJDIR)/Makedepend $(OBJDIR)/*.d
	-rm -rf $(OBJDIR)/ii_files
	-rmdir $(OBJDIR)

spotless : clean
	-rm -f $(PROGS)
	-rmdir $(DESTDIR)

