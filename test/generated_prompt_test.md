You are an expert embedded software engineer specializing in automotive C development.

Your task is to answer questions about the provided source code.

The provided files represent part of an automotive software project. Analyze the code and understand:
- The purpose of each module.
- Relationships between source and header files.
- Functions, macros, global variables, and data structures.
- Control flow.
- APIs between modules.
- Configuration values.
- Hardware interactions when present.

When answering:
- Base your answer only on the provided code.
- If information is not available, explicitly state that it cannot be determined.
- Do not invent functions or behavior.
- Reference the relevant file(s) and function(s) that support your answer.
- Explain your reasoning when appropriate.

========================
File: D:\GitProjects\embedded-ai-copilot\sample_project\inc\app.h
========================
#ifndef APP_H
#define APP_H

void App_Init(void);
void App_Run(void);

#endif
========================
File: D:\GitProjects\embedded-ai-copilot\sample_project\inc\can.h
========================
#ifndef CAN_H
#define CAN_H

#include <stdint.h>

typedef struct
{
    uint32_t id;
    uint8_t dlc;
    uint8_t data[8];
} Can_Frame;

void Can_Init(void);
void Can_SendFrame(const Can_Frame *frame);

#endif
========================
File: D:\GitProjects\embedded-ai-copilot\sample_project\inc\gpio.h
========================
#ifndef GPIO_H
#define GPIO_H

void Gpio_Init(void);
void Gpio_ToggleLed(void);

#endif
========================
File: D:\GitProjects\embedded-ai-copilot\sample_project\src\app.c
========================
#include "app.h"
#include "can.h"
#include "gpio.h"

void App_Init(void)
{
    Gpio_Init();
    Can_Init();
}

void App_Run(void)
{
    Can_Frame frame =
    {
        .id = 0x123,
        .dlc = 2,
        .data = {0xAA, 0x55}
    };

    Can_SendFrame(&frame);
    Gpio_ToggleLed();
}
========================
File: D:\GitProjects\embedded-ai-copilot\sample_project\src\can.c
========================
#include "can.h"
#include <stdio.h>

static int can_initialized = 0;

void Can_Init(void)
{
    printf("Initializing CAN peripheral...\n");

    /* Configure baud rate */
    /* Configure filters */
    /* Enable interrupts */

    can_initialized = 1;
}

void Can_SendFrame(const Can_Frame *frame)
{
    if (!can_initialized)
    {
        return;
    }

    printf("Sending CAN frame with ID: %lu\n",
           (unsigned long)frame->id);
}
========================
File: D:\GitProjects\embedded-ai-copilot\sample_project\src\gpio.c
========================
#include "gpio.h"
#include <stdio.h>

void Gpio_Init(void)
{
    printf("GPIO initialized\n");
}

void Gpio_ToggleLed(void)
{
    printf("LED toggled\n");
}
========================
File: D:\GitProjects\embedded-ai-copilot\sample_project\src\main.c
========================
#include "app.h"

int main(void)
{
    App_Init();

    while (1)
    {
        App_Run();
    }

    return 0;
}

Question:
Can you explain to me what the can module does?