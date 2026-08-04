PROJECT := cema_h563_bench
MODEL_ID ?= nmc_g4_dst_s0
BUILD_DIR := Build/$(MODEL_ID)

TOOLCHAIN ?= /opt/st/stm32cubeide_2.2.0/plugins/com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.rel1.linux64_1.0.100.202602081740/tools/bin
CC := $(TOOLCHAIN)/arm-none-eabi-gcc
OBJCOPY := $(TOOLCHAIN)/arm-none-eabi-objcopy
OBJDUMP := $(TOOLCHAIN)/arm-none-eabi-objdump
SIZE := $(TOOLCHAIN)/arm-none-eabi-size

FW_ROOT ?= $(HOME)/STM32Cube/Repository/STM32Cube_FW_H5_V1.7.0
HAL_SRC := $(FW_ROOT)/Drivers/STM32H5xx_HAL_Driver/Src
HAL_INC := $(FW_ROOT)/Drivers/STM32H5xx_HAL_Driver/Inc
CMSIS_DEVICE := $(FW_ROOT)/Drivers/CMSIS/Device/ST/STM32H5xx/Include
CMSIS_CORE := $(FW_ROOT)/Drivers/CMSIS/Include
AI_INC := AI/Runtime/Inc
AI_LIB := AI/Runtime/Lib/NetworkRuntime1201_CM33_GCC.a

CFLAGS := -mcpu=cortex-m33 -mthumb -mfpu=fpv5-sp-d16 -mfloat-abi=hard
CFLAGS += -std=gnu11 -Os -ffunction-sections -fdata-sections -Wall
CFLAGS += -fstack-usage -MMD -MP --specs=nano.specs
CFLAGS += -DUSE_HAL_DRIVER -DSTM32H563xx
CFLAGS += -IInc -I$(HAL_INC) -I$(HAL_INC)/Legacy -I$(CMSIS_DEVICE) -I$(CMSIS_CORE)
CFLAGS += -I$(AI_INC) -IAI/Generated
ASFLAGS := $(CFLAGS) -x assembler-with-cpp
LDFLAGS := -mcpu=cortex-m33 -mthumb -mfpu=fpv5-sp-d16 -mfloat-abi=hard
LDFLAGS += -TSTM32H563ZITX_FLASH.ld --specs=nosys.specs --specs=nano.specs
LDFLAGS += -Wl,-Map=$(BUILD_DIR)/$(PROJECT).map -Wl,--gc-sections -static

APP_SRCS := \
	Src/main.c \
	Src/stm32h5xx_hal_msp.c \
	Src/stm32h5xx_it.c \
	Src/syscalls.c \
	Src/sysmem.c \
	Src/system_stm32h5xx.c \
	AI/Generated/cema.c \
	AI/Generated/cema_data.c

HAL_SRCS := \
	$(HAL_SRC)/stm32h5xx_hal.c \
	$(HAL_SRC)/stm32h5xx_hal_cortex.c \
	$(HAL_SRC)/stm32h5xx_hal_dma.c \
	$(HAL_SRC)/stm32h5xx_hal_dma_ex.c \
	$(HAL_SRC)/stm32h5xx_hal_exti.c \
	$(HAL_SRC)/stm32h5xx_hal_flash.c \
	$(HAL_SRC)/stm32h5xx_hal_flash_ex.c \
	$(HAL_SRC)/stm32h5xx_hal_gpio.c \
	$(HAL_SRC)/stm32h5xx_hal_icache.c \
	$(HAL_SRC)/stm32h5xx_hal_pwr.c \
	$(HAL_SRC)/stm32h5xx_hal_pwr_ex.c \
	$(HAL_SRC)/stm32h5xx_hal_rcc.c \
	$(HAL_SRC)/stm32h5xx_hal_rcc_ex.c \
	$(HAL_SRC)/stm32h5xx_hal_uart.c \
	$(HAL_SRC)/stm32h5xx_hal_uart_ex.c

APP_OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(APP_SRCS))
HAL_OBJS := $(patsubst $(HAL_SRC)/%.c,$(BUILD_DIR)/HAL/%.o,$(HAL_SRCS))
STARTUP_OBJ := $(BUILD_DIR)/Startup/startup_stm32h563zitx.o
OBJS := $(APP_OBJS) $(HAL_OBJS) $(STARTUP_OBJ)
DEPS := $(OBJS:.o=.d)
ELF := $(BUILD_DIR)/$(PROJECT).elf

.PHONY: all clean size
all: $(ELF) $(BUILD_DIR)/$(PROJECT).bin $(BUILD_DIR)/$(PROJECT).hex $(BUILD_DIR)/$(PROJECT).list
	$(SIZE) $(ELF) | tee $(BUILD_DIR)/size.txt

$(ELF): $(OBJS) $(AI_LIB) STM32H563ZITX_FLASH.ld
	@mkdir -p $(dir $@)
	$(CC) $(OBJS) $(AI_LIB) $(LDFLAGS) -Wl,--start-group -lc -lm -Wl,--end-group -o $@

$(BUILD_DIR)/$(PROJECT).bin: $(ELF)
	$(OBJCOPY) -O binary $< $@

$(BUILD_DIR)/$(PROJECT).hex: $(ELF)
	$(OBJCOPY) -O ihex $< $@

$(BUILD_DIR)/$(PROJECT).list: $(ELF)
	$(OBJDUMP) -h -S $< > $@

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/HAL/%.o: $(HAL_SRC)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(STARTUP_OBJ): Startup/startup_stm32h563zitx.s
	@mkdir -p $(dir $@)
	$(CC) $(ASFLAGS) -c $< -o $@

clean:
	rm -rf Build

-include $(DEPS)
