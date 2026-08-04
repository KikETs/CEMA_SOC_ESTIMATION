#ifndef CEMA_KF_CORE_H
#define CEMA_KF_CORE_H

#include <stdint.h>

typedef struct
{
  double initial_soc;
  double initial_voltage_v;
  double initial_temperature_c;
  double nominal_temperature_c;
} CEMA_KF_Reset;

typedef struct
{
  double voltage_v;
  double current_a;
  double temperature_c;
  double dt_s;
} CEMA_KF_Sample_Dt;

void CEMA_KF_Reset_State(const CEMA_KF_Reset *reset);
double CEMA_KF_Step(
    double voltage_v,
    double raw_current_a,
    double temperature_c,
    double dt_s);
uint32_t CEMA_KF_Status(void);
uint32_t CEMA_KF_Chemistry(void);
uint32_t CEMA_KF_Method(void);
uint32_t CEMA_KF_State_Dim(void);
uint32_t CEMA_KF_Asset_Bytes(void);
uint32_t CEMA_KF_Runtime_State_Bytes(void);
uint32_t CEMA_KF_Sample_Count(void);
const char *CEMA_KF_Model_Id(void);

#endif
