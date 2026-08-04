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
#include "preprocess_config.h"
#include "stai.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define CEMA_PROTOCOL_MAGIC 0x43454D41UL
#define CEMA_PROTOCOL_VERSION 2UL
#define CEMA_CMD_QUERY 0x51U
#define CEMA_CMD_RESET 0x52U
#define CEMA_CMD_SAMPLE 0x53U
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
static uint32_t *g_stack_paint_end;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART3_UART_Init(void);
static void MX_ICACHE_Init(void);
/* USER CODE BEGIN PFP */
static void CEMA_DWT_Init(void);
static stai_return_code CEMA_Model_Init(void);
static void CEMA_Preprocess_Reset(void);
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
  float voltage_v;
  float current_a;
  float temperature_c;
} CEMA_Raw_Sample;

typedef struct __attribute__((packed))
{
  uint32_t magic;
  uint32_t status;
  uint32_t ready;
  uint32_t total_cycles;
  uint32_t network_cycles;
  uint32_t stack_highwater_bytes;
  uint32_t sample_count;
  float output;
} CEMA_Stream_Response;

typedef struct
{
  double vcorr;
  double vcorr_ema50;
  double vcorr_ema200;
  double vcorr_ema800;
  double current_ema50;
  double current_ema200;
  double absi_ema50;
  double absi_ema200;
  uint32_t sample_count;
} CEMA_Preprocess_State;

static CEMA_Preprocess_State g_preprocess;

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

static double CEMA_R0_For_Temperature(float temperature_c)
{
  uint32_t index;
  const double temperature = (double)temperature_c;

  if (temperature <= CEMA_R0_TEMPERATURES[0])
  {
    return CEMA_R0_VALUES[0];
  }
  if (temperature >= CEMA_R0_TEMPERATURES[CEMA_R0_COUNT - 1U])
  {
    return CEMA_R0_VALUES[CEMA_R0_COUNT - 1U];
  }
  for (index = 1U; index < CEMA_R0_COUNT; ++index)
  {
    if (temperature <= CEMA_R0_TEMPERATURES[index])
    {
      const double x0 = CEMA_R0_TEMPERATURES[index - 1U];
      const double x1 = CEMA_R0_TEMPERATURES[index];
      const double y0 = CEMA_R0_VALUES[index - 1U];
      const double y1 = CEMA_R0_VALUES[index];
      const double fraction = (temperature - x0) / (x1 - x0);
      return y0 + fraction * (y1 - y0);
    }
  }
  return CEMA_R0_VALUES[CEMA_R0_COUNT - 1U];
}

static void CEMA_Preprocess_Reset(void)
{
  memset(&g_preprocess, 0, sizeof(g_preprocess));
  memset(g_input_window, 0, sizeof(g_input_window));
}

static void CEMA_Update_EMA(double *state, float input, double alpha)
{
  *state = alpha * (*state) + (1.0 - alpha) * (double)input;
}

static uint32_t CEMA_Preprocess_Sample(const CEMA_Raw_Sample *sample)
{
  float all_features[CEMA_ALL_FEATURE_COUNT];
  float selected[CEMA_CHANNEL_COUNT];
  float v_corr;
  float current = sample->current_a;
  float abs_current = (current < 0.0F) ? -current : current;
  float v_drop;
  double v_removed;
  uint32_t channel;

  v_drop = (float)((double)current *
                   CEMA_R0_For_Temperature(sample->temperature_c));
  v_removed = (double)sample->voltage_v - (double)v_drop;

  if (g_preprocess.sample_count == 0U)
  {
    g_preprocess.vcorr = v_removed;
    v_corr = (float)g_preprocess.vcorr;
    g_preprocess.vcorr_ema50 = (double)v_corr;
    g_preprocess.vcorr_ema200 = (double)v_corr;
    g_preprocess.vcorr_ema800 = (double)v_corr;
    g_preprocess.current_ema50 = (double)current;
    g_preprocess.current_ema200 = (double)current;
    g_preprocess.absi_ema50 = (double)abs_current;
    g_preprocess.absi_ema200 = (double)abs_current;
  }
  else
  {
    g_preprocess.vcorr =
        CEMA_ALPHA_VCORR * g_preprocess.vcorr +
        (1.0 - CEMA_ALPHA_VCORR) * v_removed;
    v_corr = (float)g_preprocess.vcorr;
    CEMA_Update_EMA(&g_preprocess.vcorr_ema50, v_corr, CEMA_ALPHA_50);
    CEMA_Update_EMA(&g_preprocess.vcorr_ema200, v_corr, CEMA_ALPHA_200);
    CEMA_Update_EMA(&g_preprocess.vcorr_ema800, v_corr, CEMA_ALPHA_800);
    CEMA_Update_EMA(&g_preprocess.current_ema50, current, CEMA_ALPHA_50);
    CEMA_Update_EMA(&g_preprocess.current_ema200, current, CEMA_ALPHA_200);
    CEMA_Update_EMA(&g_preprocess.absi_ema50, abs_current, CEMA_ALPHA_50);
    CEMA_Update_EMA(&g_preprocess.absi_ema200, abs_current, CEMA_ALPHA_200);
  }

  all_features[0] = v_corr;
  all_features[1] = current;
  all_features[2] = sample->temperature_c;
  all_features[3] = (float)g_preprocess.vcorr_ema50;
  all_features[4] = v_corr - all_features[3];
  all_features[5] = (float)g_preprocess.vcorr_ema200;
  all_features[6] = v_corr - all_features[5];
  all_features[7] = (float)g_preprocess.vcorr_ema800;
  all_features[8] = v_corr - all_features[7];
  all_features[9] = (float)g_preprocess.current_ema50;
  all_features[10] = current - all_features[9];
  all_features[11] = (float)g_preprocess.current_ema200;
  all_features[12] = current - all_features[11];
  all_features[13] = (float)g_preprocess.absi_ema50;
  all_features[14] = abs_current - all_features[13];
  all_features[15] = (float)g_preprocess.absi_ema200;
  all_features[16] = abs_current - all_features[15];

  for (channel = 0U; channel < CEMA_CHANNEL_COUNT; ++channel)
  {
    const float value = all_features[CEMA_CHANNEL_IDS[channel]];
    selected[channel] =
        (value - CEMA_SCALER_MEAN[channel]) / CEMA_SCALER_STD[channel];
  }

  if (g_preprocess.sample_count < CEMA_WINDOW_SIZE)
  {
    memcpy(
        &g_input_window[g_preprocess.sample_count * CEMA_CHANNEL_COUNT *
                        sizeof(float)],
        selected,
        CEMA_CHANNEL_COUNT * sizeof(float));
  }
  else
  {
    memmove(
        g_input_window,
        &g_input_window[CEMA_CHANNEL_COUNT * sizeof(float)],
        (CEMA_WINDOW_SIZE - 1U) * CEMA_CHANNEL_COUNT * sizeof(float));
    memcpy(
        &g_input_window[(CEMA_WINDOW_SIZE - 1U) * CEMA_CHANNEL_COUNT *
                        sizeof(float)],
        selected,
        CEMA_CHANNEL_COUNT * sizeof(float));
  }
  ++g_preprocess.sample_count;
  return (g_preprocess.sample_count >= CEMA_WINDOW_SIZE) ? 1U : 0U;
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
  const uint32_t response[8] = {
      CEMA_PROTOCOL_MAGIC,
      CEMA_PROTOCOL_VERSION,
      SystemCoreClock,
      3U,
      sizeof(CEMA_Raw_Sample),
      STAI_CEMA_MACC_NUM,
      CEMA_CHANNEL_COUNT,
      CEMA_WINDOW_SIZE};
  UART_Send(response, (uint16_t)sizeof(response));
}

static void CEMA_Send_Reset(void)
{
  const uint32_t response[2] = {CEMA_PROTOCOL_MAGIC, g_model_status};
  CEMA_Preprocess_Reset();
  UART_Send(response, (uint16_t)sizeof(response));
}

static void CEMA_Send_Sample(const CEMA_Raw_Sample *sample)
{
  CEMA_Stream_Response response = {0};
  uint32_t primask;
  uint32_t network_start;

  response.magic = CEMA_PROTOCOL_MAGIC;
  response.status = g_model_status;
  if (g_model_status != (uint32_t)STAI_SUCCESS)
  {
    UART_Send(&response, (uint16_t)sizeof(response));
    return;
  }

  primask = __get_PRIMASK();
  __disable_irq();
  __DSB();
  __ISB();
  DWT->CYCCNT = 0UL;
  response.ready = CEMA_Preprocess_Sample(sample);
  if (response.ready != 0U)
  {
    memcpy(g_input, g_input_window, STAI_CEMA_IN_1_SIZE_BYTES);
    __DSB();
    __ISB();
    network_start = DWT->CYCCNT;
    g_model_status = (uint32_t)stai_cema_run(g_network, STAI_MODE_SYNC);
    __DSB();
    __ISB();
    response.network_cycles = DWT->CYCCNT - network_start;
  }
  response.total_cycles = DWT->CYCCNT;
  if (primask == 0U)
  {
    __enable_irq();
  }
  response.status = g_model_status;
  response.stack_highwater_bytes = Stack_Highwater_Bytes();
  response.sample_count = g_preprocess.sample_count;
  if ((response.ready != 0U) && (g_output != NULL))
  {
    memcpy(&response.output, g_output, sizeof(response.output));
  }
  UART_Send(&response, (uint16_t)sizeof(response));
}

static void CEMA_Protocol_Loop(void)
{
  uint8_t command;
  CEMA_Raw_Sample sample;

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
    else if (command == CEMA_CMD_RESET)
    {
      CEMA_Send_Reset();
    }
    else if (command == CEMA_CMD_SAMPLE)
    {
      if (HAL_UART_Receive(
              &huart3,
              (uint8_t *)&sample,
              sizeof(sample),
              HAL_MAX_DELAY) == HAL_OK)
      {
        CEMA_Send_Sample(&sample);
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
  CEMA_Preprocess_Reset();
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
