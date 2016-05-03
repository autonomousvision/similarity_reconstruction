LIBS = -lm

CFLAGS = -g -O3

CC = /usr/bin/g++
C = /usr/bin/gcc

.SUFFIXES: .o .cpp .c

.cpp.o: 
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $*.o $<

.c.o: 
	$(C) $(CFLAGS) $(INCLUDE) -c -o $*.o $<

ALL.O = trimesh.o plyfile.o ply2vri.o vrip/ChunkAllocator.o vrip/ChunkLink.o \
	vrip/OccGrid.o vrip/OccGridRLE.o

ply2vri: $(ALL.O)
	$(CC) $(CFLAGS) -o $@ $(ALL.O) $(INCLUDE) $(LIBDIR) $(LIBS)

clean:
	rm *.o vrip/*.o
