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