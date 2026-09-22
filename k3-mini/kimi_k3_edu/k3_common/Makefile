# K3-Edu Inference CLI
CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c99 -D_POSIX_C_SOURCE=200809L
LDFLAGS = -lm

SRCS = k3_main.c k3_common.c k3_tensor.c k3_model.c k3_tokenizer.c k3_sampler.c k3_tui.c
OBJS = $(SRCS:.c=.o)
TARGET = k3_infer

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
