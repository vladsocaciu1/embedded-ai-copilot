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