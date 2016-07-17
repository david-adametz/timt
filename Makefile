CC=g++
CFLAGS=-O3 -c -Wall -I/armadillo/usr/local/include/ -I/boost_1_55_0
LDFLAGS=-L/usr/local/lib/ -L/boost_1_55_0/stage/lib/ -lqrupdate /usr/local/lib/libarmadillo.so.4.300.5 -lboost_random
SOURCES=src/timt.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=timt

all: $(SOURCES) $(EXECUTABLE)
		
$(EXECUTABLE): $(OBJECTS) 
		$(CC) $(OBJECTS) -o $@ $(LDFLAGS) 

.cpp.o:
		$(CC) $(CFLAGS) $< -o $@

clean:
   rm -f *.o timt
