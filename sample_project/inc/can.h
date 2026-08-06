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