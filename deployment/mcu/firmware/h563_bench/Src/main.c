/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <string.h>
#include "cema.h"
#include "model_config.h"
#include "stai.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define CEMA_PROTOCOL_MAGIC 0x43454D41UL
#define CEMA_PROTOCOL_VERSION 1UL
#define CEMA_CMD_QUERY 0x51U
#define CEMA_CMD_WINDOW 0x57U
#define CEMA_CMD_BENCH 0x42U
#define CEMA_MAX_REPS 2048U
#define CEMA_WARMUP_REPS 10U
#define STACK_PATTERN 0xA5A5A5A5UL
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

UART_HandleTypeDef huart3;

/* USER CODE BEGIN PV */
STAI_NETWORK_CONTEXT_DECLARE(g_network, STAI_CEMA_CONTEXT_SIZE)
STAI_ALIGNED(STAI_CEMA_ACTIVATION_1_ALIGNMENT)
static uint8_t g_activations[STAI_CEMA_ACTIVATION_1_SIZE_BYTES];
STAI_ALIGNED(4)
static uint8_t g_input_window[STAI_CEMA_IN_1_SIZE_BYTES];
static stai_ptr g_input;
static stai_ptr g_output;
static uint32_t g_model_status;
static uint32_t g_cycles[CEMA_MAX_REPS];
static uint32_t *g_stack_paint_end;
static uint8_t g_benchmark_warmed;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART3_UART_Init(void);
static void MX_ICACHE_Init(void);
/* USER CODE BEGIN PFP */
static void CEMA_DWT_Init(void);
static stai_return_code CEMA_Model_Init(void);
static uint32_t CEMA_Run_Once(void);
static void CEMA_Protocol_Loop(void);
static void Stack_Watermark_Init(void);
static uint32_t Stack_Highwater_Bytes(void);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
extern uint32_t _sstack;
extern uint32_t _estack;

typedef struct __attribute__((packed))
{
  uint32_t magic;
  uint32_t status;
  uint32_t value0;
  uint32_t value1;
  float output;
} CEMA_Response;

static void UART_Send(const void *data, uint16_t size)
{
  (void)HAL_UART_Transmit(&huart3, (const uint8_t *)data, size, HAL_MAX_DELAY);
}

static void CEMA_DWT_Init(void)
{
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
#if defined(DWT_LAR)
  DWT->LAR = 0xC5ACCE55UL;
#endif
  DWT->CYCCNT = 0UL;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
  __DSB();
  __ISB();
}

static stai_return_code CEMA_Model_Init(void)
{
  stai_return_code result;
  stai_ptr activation_buffers[STAI_CEMA_ACTIVATIONS_NUM] = {g_activations};
  stai_ptr input_buffers[STAI_CEMA_IN_NUM] = {0};
  stai_ptr output_buffers[STAI_CEMA_OUT_NUM] = {0};
  stai_size count = 0;

  result = stai_runtime_init();
  if (result != STAI_SUCCESS)
  {
    return result;
  }
  result = stai_cema_init(g_network);
  if (result != STAI_SUCCESS)
  {
    return result;
  }
  result = stai_cema_set_activations(
      g_network, activation_buffers, STAI_CEMA_ACTIVATIONS_NUM);
  if (result != STAI_SUCCESS)
  {
    return result;
  }
  result = stai_cema_get_inputs(g_network, input_buffers, &count);
  if ((result != STAI_SUCCESS) || (count != STAI_CEMA_IN_NUM))
  {
    return (result != STAI_SUCCESS) ? result : STAI_ERROR_NETWORK_INVALID_IN_NUM;
  }
  g_input = input_buffers[0];
  result = stai_cema_get_outputs(g_network, output_buffers, &count);
  if ((result != STAI_SUCCESS) || (count != STAI_CEMA_OUT_NUM))
  {
    return (result != STAI_SUCCESS) ? result : STAI_ERROR_NETWORK_INVALID_OUT_NUM;
  }
  g_output = output_buffers[0];
  return STAI_SUCCESS;
}

static uint32_t CEMA_Run_Once(void)
{
  uint32_t cycles;

  /*
   * ST Edge AI may place the input tensor inside the reusable activation arena.
   * Restore the immutable benchmark window before every run. This copy is
   * intentionally outside the timed region.
   */
  memcpy(g_input, g_input_window, STAI_CEMA_IN_1_SIZE_BYTES);
  __DSB();
  __ISB();
  DWT->CYCCNT = 0UL;
  __DSB();
  __ISB();
  g_model_status = (uint32_t)stai_cema_run(g_network, STAI_MODE_SYNC);
  __DSB();
  __ISB();
  cycles = DWT->CYCCNT;
  return cycles;
}

static void Stack_Watermark_Init(void)
{
  uintptr_t stack_pointer;
  uint32_t *cursor;

  __asm volatile ("mrs %0, msp" : "=r" (stack_pointer));
  if (stack_pointer > ((uintptr_t)&_sstack + 512U))
  {
    g_stack_paint_end = (uint32_t *)(stack_pointer - 256U);
    for (cursor = &_sstack; cursor < g_stack_paint_end; ++cursor)
    {
      *cursor = STACK_PATTERN;
    }
  }
  else
  {
    g_stack_paint_end = &_sstack;
  }
}

static uint32_t Stack_Highwater_Bytes(void)
{
  uint32_t *cursor;
  uint32_t *lowest_used = g_stack_paint_end;

  for (cursor = &_sstack; cursor < g_stack_paint_end; ++cursor)
  {
    if (*cursor != STACK_PATTERN)
    {
      lowest_used = cursor;
      break;
    }
  }
  return (uint32_t)((uintptr_t)&_estack - (uintptr_t)lowest_used);
}

static void CEMA_Send_Query(void)
{
  const uint32_t response[6] = {
      CEMA_PROTOCOL_MAGIC,
      CEMA_PROTOCOL_VERSION,
      SystemCoreClock,
      STAI_CEMA_IN_1_SIZE,
      STAI_CEMA_IN_1_SIZE_BYTES,
      STAI_CEMA_MACC_NUM};
  UART_Send(response, (uint16_t)sizeof(response));
}

static void CEMA_Send_Inference(void)
{
  CEMA_Response response = {0};

  if (g_model_status == (uint32_t)STAI_SUCCESS)
  {
    response.value0 = CEMA_Run_Once();
  }
  response.magic = CEMA_PROTOCOL_MAGIC;
  response.status = g_model_status;
  response.value1 = Stack_Highwater_Bytes();
  if (g_output != NULL)
  {
    memcpy(&response.output, g_output, sizeof(response.output));
  }
  UART_Send(&response, (uint16_t)sizeof(response));
}

static void CEMA_Send_Benchmark(uint32_t repetitions)
{
  CEMA_Response response = {0};
  uint32_t index;
  uint32_t primask;

  response.magic = CEMA_PROTOCOL_MAGIC;
  response.status = g_model_status;
  response.value0 = repetitions;
  response.value1 = SystemCoreClock;
  if ((g_model_status != (uint32_t)STAI_SUCCESS) ||
      (repetitions == 0U) ||
      (repetitions > CEMA_MAX_REPS))
  {
    response.status = (uint32_t)STAI_ERROR_NETWORK_INVALID_RUN;
    UART_Send(&response, (uint16_t)sizeof(response));
    return;
  }

  for (index = 0U; (g_benchmark_warmed == 0U) &&
                       (index < CEMA_WARMUP_REPS); ++index)
  {
    (void)CEMA_Run_Once();
  }
  g_benchmark_warmed = 1U;
  primask = __get_PRIMASK();
  __disable_irq();
  for (index = 0U; index < repetitions; ++index)
  {
    g_cycles[index] = CEMA_Run_Once();
  }
  if (primask == 0U)
  {
    __enable_irq();
  }
  response.status = g_model_status;
  if (g_output != NULL)
  {
    memcpy(&response.output, g_output, sizeof(response.output));
  }
  UART_Send(&response, (uint16_t)sizeof(response));
  UART_Send(g_cycles, (uint16_t)(repetitions * sizeof(g_cycles[0])));
}

static void CEMA_Protocol_Loop(void)
{
  uint8_t command;
  uint32_t repetitions;

  for (;;)
  {
    if (HAL_UART_Receive(&huart3, &command, 1U, HAL_MAX_DELAY) != HAL_OK)
    {
      continue;
    }
    if (command == CEMA_CMD_QUERY)
    {
      CEMA_Send_Query();
    }
    else if (command == CEMA_CMD_WINDOW)
    {
      if (HAL_UART_Receive(
              &huart3,
              g_input_window,
              STAI_CEMA_IN_1_SIZE_BYTES,
              HAL_MAX_DELAY) == HAL_OK)
      {
        CEMA_Send_Inference();
      }
    }
    else if (command == CEMA_CMD_BENCH)
    {
      if (HAL_UART_Receive(
              &huart3,
              (uint8_t *)&repetitions,
              sizeof(repetitions),
              HAL_MAX_DELAY) == HAL_OK)
      {
        CEMA_Send_Benchmark(repetitions);
      }
    }
  }
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART3_UART_Init();
  MX_ICACHE_Init();
  /* USER CODE BEGIN 2 */
  CEMA_DWT_Init();
  g_model_status = (uint32_t)CEMA_Model_Init();
  Stack_Watermark_Init();
  HAL_GPIO_WritePin(LED1_GPIO_Port, LED1_Pin, GPIO_PIN_SET);
  CEMA_Protocol_Loop();
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE0);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_BYPASS_DIGITAL;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLL1_SOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 250;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1_VCIRANGE_1;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1_VCORANGE_WIDE;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_PCLK3;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure the programming delay
  */
  __HAL_FLASH_SET_PROGRAM_DELAY(FLASH_PROGRAMMING_DELAY_2);
}

/**
  * @brief ICACHE Initialization Function
  * @param None
  * @retval None
  */
static void MX_ICACHE_Init(void)
{

  /* USER CODE BEGIN ICACHE_Init 0 */

  /* USER CODE END ICACHE_Init 0 */

  /* USER CODE BEGIN ICACHE_Init 1 */

  /* USER CODE END ICACHE_Init 1 */

  /** Enable instruction cache in 1-way (direct mapped cache)
  */
  if (HAL_ICACHE_ConfigAssociativityMode(ICACHE_1WAY) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_ICACHE_Enable() != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ICACHE_Init 2 */

  /* USER CODE END ICACHE_Init 2 */

}

/**
  * @brief USART3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART3_UART_Init(void)
{

  /* USER CODE BEGIN USART3_Init 0 */

  /* USER CODE END USART3_Init 0 */

  /* USER CODE BEGIN USART3_Init 1 */

  /* USER CODE END USART3_Init 1 */
  huart3.Instance = USART3;
  huart3.Init.BaudRate = 921600;
  huart3.Init.WordLength = UART_WORDLENGTH_8B;
  huart3.Init.StopBits = UART_STOPBITS_1;
  huart3.Init.Parity = UART_PARITY_NONE;
  huart3.Init.Mode = UART_MODE_TX_RX;
  huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart3.Init.OverSampling = UART_OVERSAMPLING_16;
  huart3.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart3.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart3.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart3, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart3, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART3_Init 2 */

  /* USER CODE END USART3_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LED1_GPIO_Port, LED1_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : USER_BUTTON_Pin */
  GPIO_InitStruct.Pin = USER_BUTTON_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(USER_BUTTON_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : LED1_Pin */
  GPIO_InitStruct.Pin = LED1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LED1_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @param None
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
