TARGET=c-main
all: $(TARGET)
%.o: %.asm
	nasm -f elf64 -g -Fdwarf $<
$(TARGET): asm-module.o c-main.o
	cc -g -std=c99 -no-pie -z noexecstack $^ -o $@
clean:
	rm -rf $(TARGET) *.o */*.o
