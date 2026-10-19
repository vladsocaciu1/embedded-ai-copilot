from pathlib import Path

import pytest

from app.code_graph import build_graph
from app.code_index import build_index
from app.context_selector import rank_files
from app.models import ProjectFile


BOARDS = ("STM32F4-Discovery", "STM32F429I-Discovery")
EXAMPLES = {
    "GPIO_EXTI": (
        "EXTI0_IRQHandler",
        "HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_0);",
        "BSP_LED_Toggle(LED3);",
    ),
    "UART_TwoBoards_ComIT": (
        "USART2_IRQHandler",
        "HAL_UART_IRQHandler(&UartHandle);",
        "HAL_UART_Transmit_IT(&UartHandle, aTxBuffer, TXBUFFERSIZE);",
    ),
    "ADC_RegularConversion": (
        "ADC_IRQHandler",
        "HAL_ADC_IRQHandler(&AdcHandle);",
        "HAL_ADC_Start_IT(&AdcHandle);",
    ),
}


def eval_files() -> list[ProjectFile]:
    files = []
    for board in BOARDS:
        for example, (irq, irq_body, main_call) in EXAMPLES.items():
            root = Path("Projects", board, "Examples", example.split("_")[0], example)
            files += [
                ProjectFile(
                    root / "Inc/main.h",
                    '#include "stm32f4xx_hal.h"\nvoid Error_Handler(void);\n',
                ),
                ProjectFile(
                    root / "Inc/stm32f4xx_it.h",
                    f"void SysTick_Handler(void);\nvoid {irq}(void);\n",
                ),
                ProjectFile(
                    root / "Src/main.c",
                    '#include "main.h"\nstatic void SystemClock_Config(void);\n'
                    "int main(void)\n{\n    HAL_Init();\n    SystemClock_Config();\n"
                    f"    BSP_LED_Init(LED3);\n    {main_call}\n    while (1) {{ }}\n}}\n"
                    "static void SystemClock_Config(void)\n{\n    HAL_RCC_OscConfig(&RCC_OscInitStruct);\n"
                    "    HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5);\n}\n"
                    "void Error_Handler(void)\n{\n    while (1) { }\n}\n",
                ),
                ProjectFile(
                    root / "Src/stm32f4xx_it.c",
                    '#include "main.h"\n#include "stm32f4xx_it.h"\n'
                    "void SysTick_Handler(void)\n{\n    HAL_IncTick();\n}\n"
                    f"void {irq}(void)\n{{\n    {irq_body}\n}}\n",
                ),
                ProjectFile(
                    root / "Src/system_stm32f4xx.c",
                    "uint32_t SystemCoreClock = 16000000;\nvoid SystemInit(void)\n{\n}\n"
                    "void SystemCoreClockUpdate(void)\n{\n}\n",
                ),
                ProjectFile(
                    root / "EWARM/startup_stm32f407xx.s",
                    "Reset_Handler\n  BL SystemInit\n  BL main\n",
                ),
            ]
    bsp = Path("Drivers/BSP/STM32F4-Discovery")
    files += [
        ProjectFile(
            bsp / "stm32f4_discovery.h",
            "void BSP_LED_Init(Led_TypeDef Led);\nvoid BSP_LED_Toggle(Led_TypeDef Led);\n",
        ),
        ProjectFile(
            bsp / "stm32f4_discovery.c",
            '#include "stm32f4_discovery.h"\n'
            "void BSP_LED_Init(Led_TypeDef Led)\n{\n    HAL_GPIO_Init(GPIO_PORT[Led], &GPIO_InitStruct);\n}\n"
            "void BSP_LED_Toggle(Led_TypeDef Led)\n{\n    HAL_GPIO_TogglePin(GPIO_PORT[Led], GPIO_PIN[Led]);\n}\n",
        ),
    ]
    return files


_FILES = eval_files()
_INDEX = build_index(_FILES)
_GRAPH = build_graph(_INDEX)

_CASES = [
    pytest.param(
        "In STM32F4-Discovery GPIO_EXTI, what does EXTI0_IRQHandler call?",
        ["Projects/STM32F4-Discovery/Examples/GPIO/GPIO_EXTI/Src/stm32f4xx_it.c"],
        1,
        id="E1",
    ),
    pytest.param(
        "How is the system clock configured in the GPIO_EXTI example for STM32F4-Discovery?",
        ["Projects/STM32F4-Discovery/Examples/GPIO/GPIO_EXTI/Src/main.c"],
        2,
        id="E2",
    ),
    pytest.param(
        "What does BSP_LED_Init do?",
        [
            "Drivers/BSP/STM32F4-Discovery/stm32f4_discovery.c",
            "Drivers/BSP/STM32F4-Discovery/stm32f4_discovery.h",
        ],
        2,
        id="E3",
    ),
    pytest.param(
        "Explain Projects/STM32F429I-Discovery/Examples/UART/UART_TwoBoards_ComIT/Src/main.c",
        ["Projects/STM32F429I-Discovery/Examples/UART/UART_TwoBoards_ComIT/Src/main.c"],
        1,
        id="E4",
    ),
    pytest.param(
        "How does UART_TwoBoards_ComIT send data on STM32F4-Discovery?",
        ["Projects/STM32F4-Discovery/Examples/UART/UART_TwoBoards_ComIT/Src/main.c"],
        3,
        id="E5",
    ),
    pytest.param(
        "Where is HAL_UART_Transmit_IT called?",
        [
            "Projects/STM32F4-Discovery/Examples/UART/UART_TwoBoards_ComIT/Src/main.c",
            "Projects/STM32F429I-Discovery/Examples/UART/UART_TwoBoards_ComIT/Src/main.c",
        ],
        2,
        id="E6",
    ),
    # weak spot: no "interrupt"->IRQ synonym
    pytest.param(
        "Which function handles the ADC interrupt in ADC_RegularConversion on STM32F429I-Discovery?",
        ["Projects/STM32F429I-Discovery/Examples/ADC/ADC_RegularConversion/Src/stm32f4xx_it.c"],
        4,
        id="E7",
    ),
    pytest.param(
        "What does SystemInit do in STM32F4-Discovery GPIO_EXTI?",
        ["Projects/STM32F4-Discovery/Examples/GPIO/GPIO_EXTI/Src/system_stm32f4xx.c"],
        1,
        id="E8",
    ),
    pytest.param(
        "Where is EXTI0_IRQHandler declared in STM32F429I-Discovery GPIO_EXTI?",
        ["Projects/STM32F429I-Discovery/Examples/GPIO/GPIO_EXTI/Inc/stm32f4xx_it.h"],
        2,
        id="E9",
    ),
    # weak spot: the definition ranks before callers
    pytest.param(
        "Who calls BSP_LED_Toggle?",
        [
            "Projects/STM32F4-Discovery/Examples/GPIO/GPIO_EXTI/Src/main.c",
            "Projects/STM32F429I-Discovery/Examples/GPIO/GPIO_EXTI/Src/main.c",
        ],
        4,
        id="E10",
    ),
]


@pytest.mark.parametrize("question,expected,k", _CASES)
def test_selection_eval(question, expected, k):
    ranked = [path.as_posix() for path in rank_files(question, _INDEX, _GRAPH)]
    if all(path in ranked[:k] for path in expected):
        return
    lines = [f"top 5: {ranked[:5]}"]
    for path in expected:
        if path in ranked:
            rank = ranked.index(path) + 1
        else:
            rank = "absent"
        lines.append(f"{path} rank {rank}")
    pytest.fail("\n".join(lines))
