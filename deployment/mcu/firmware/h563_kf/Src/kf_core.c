#include "kf_core.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

#include "kf_config.h"

#define CEMA_MAX_STATE 4U
#define CEMA_MAX_SIGMA 9U
#define CEMA_STATUS_OK 0U
#define CEMA_STATUS_NUMERIC 1U

#if CEMA_INTERNAL_FP64
typedef double CEMA_Real;
typedef double CEMA_AssetReal;
#define CEMA_REAL_MAX DBL_MAX
#define cema_abs fabs
#define cema_atan2 atan2
#define cema_cos cos
#define cema_exp exp
#define cema_pow pow
#define cema_sin sin
#define cema_sqrt sqrt
#else
typedef float CEMA_Real;
typedef float CEMA_AssetReal;
#define CEMA_REAL_MAX FLT_MAX
#define cema_abs fabsf
#define cema_atan2 atan2f
#define cema_cos cosf
#define cema_exp expf
#define cema_pow powf
#define cema_sin sinf
#define cema_sqrt sqrtf
#endif

typedef struct
{
  CEMA_Real x[CEMA_MAX_STATE];
  CEMA_Real p[CEMA_MAX_STATE][CEMA_MAX_STATE];
  CEMA_Real previous_current_a;
  CEMA_Real adaptive_r;
  CEMA_Real q_ref_ah;
  CEMA_Real gamma;
  uint32_t sample_count;
  uint32_t status;
} CEMA_KF_State;

static CEMA_KF_State g_state;

static CEMA_Real clamp_real(CEMA_Real value, CEMA_Real low, CEMA_Real high)
{
  return fmin(high, fmax(low, value));
}

static CEMA_Real interp_temperature(
    const CEMA_AssetReal *values, CEMA_Real temperature_c)
{
  uint32_t index;
  if (temperature_c <= CEMA_TEMPERATURES[0])
  {
    return values[0];
  }
  if (temperature_c >= CEMA_TEMPERATURES[CEMA_TEMP_COUNT - 1U])
  {
    return values[CEMA_TEMP_COUNT - 1U];
  }
  for (index = 1U; index < CEMA_TEMP_COUNT; ++index)
  {
    if (temperature_c <= CEMA_TEMPERATURES[index])
    {
      const CEMA_Real x0 = CEMA_TEMPERATURES[index - 1U];
      const CEMA_Real x1 = CEMA_TEMPERATURES[index];
      const CEMA_Real weight = (temperature_c - x0) / (x1 - x0);
      return values[index - 1U] +
             weight * (values[index] - values[index - 1U]);
    }
  }
  return values[CEMA_TEMP_COUNT - 1U];
}

static CEMA_Real nearest_temperature(
    const CEMA_AssetReal *values, CEMA_Real temperature_c)
{
  uint32_t best = 0U;
  CEMA_Real best_distance =
      cema_abs(temperature_c - CEMA_TEMPERATURES[0]);
  uint32_t index;
  for (index = 1U; index < CEMA_TEMP_COUNT; ++index)
  {
    const CEMA_Real distance =
        cema_abs(temperature_c - CEMA_TEMPERATURES[index]);
    if (distance < best_distance)
    {
      best = index;
      best_distance = distance;
    }
  }
  return values[best];
}

#if CEMA_METHOD != CEMA_METHOD_CC
#if CEMA_CHEMISTRY == CEMA_CHEMISTRY_NMC
static CEMA_Real nmc_pchip_row(uint32_t temperature_index,
                           CEMA_Real soc,
                           int derivative)
{
  const CEMA_Real clipped_soc = clamp_real(soc, 0.0, 1.0);
  const CEMA_AssetReal *knots =
      CEMA_OCV_PCHIP_KNOTS[temperature_index];
  uint32_t low = 0U;
  uint32_t high = CEMA_PCHIP_SEGMENT_COUNT;
  uint32_t segment;
  CEMA_Real dx;
  CEMA_Real c0;
  CEMA_Real c1;
  CEMA_Real c2;
  CEMA_Real c3;

  if (clipped_soc >= knots[CEMA_PCHIP_SEGMENT_COUNT])
  {
    segment = CEMA_PCHIP_SEGMENT_COUNT - 1U;
  }
  else
  {
    while (high - low > 1U)
    {
      const uint32_t middle = low + (high - low) / 2U;
      if (clipped_soc < knots[middle])
      {
        high = middle;
      }
      else
      {
        low = middle;
      }
    }
    segment = low;
  }
  dx = clipped_soc - knots[segment];
  c0 = CEMA_OCV_PCHIP_COEFFICIENTS[temperature_index][0][segment];
  c1 = CEMA_OCV_PCHIP_COEFFICIENTS[temperature_index][1][segment];
  c2 = CEMA_OCV_PCHIP_COEFFICIENTS[temperature_index][2][segment];
  c3 = CEMA_OCV_PCHIP_COEFFICIENTS[temperature_index][3][segment];
  if (derivative)
  {
    return (3.0F * c0 * dx + 2.0F * c1) * dx + c2;
  }
  return ((c0 * dx + c1) * dx + c2) * dx + c3;
}

static CEMA_Real nmc_pchip_eval(
    CEMA_Real soc, CEMA_Real temperature_c, int derivative)
{
  uint32_t index;
  if (temperature_c <= CEMA_TEMPERATURES[0])
  {
    return nmc_pchip_row(0U, soc, derivative);
  }
  for (index = 1U; index < CEMA_TEMP_COUNT; ++index)
  {
    if (temperature_c <= CEMA_TEMPERATURES[index])
    {
      const CEMA_Real lower_temperature = CEMA_TEMPERATURES[index - 1U];
      const CEMA_Real upper_temperature = CEMA_TEMPERATURES[index];
      const CEMA_Real weight =
          (temperature_c - lower_temperature) /
          (upper_temperature - lower_temperature);
      const CEMA_Real lower =
          nmc_pchip_row(index - 1U, soc, derivative);
      const CEMA_Real upper = nmc_pchip_row(index, soc, derivative);
      return lower + weight * (upper - lower);
    }
  }
  return nmc_pchip_row(CEMA_TEMP_COUNT - 1U, soc, derivative);
}
#endif

static CEMA_Real grid_row(const CEMA_AssetReal *row, CEMA_Real soc)
{
  const CEMA_Real position =
      clamp_real(soc, 0.0, 1.0) * (CEMA_Real)(CEMA_SOC_COUNT - 1U);
  uint32_t lower = (uint32_t)position;
  CEMA_Real fraction;
  if (lower >= CEMA_SOC_COUNT - 1U)
  {
    return row[CEMA_SOC_COUNT - 1U];
  }
  fraction = position - (CEMA_Real)lower;
  return row[lower] + fraction * (row[lower + 1U] - row[lower]);
}

static CEMA_Real grid_eval(
                       const CEMA_AssetReal
                           grid[CEMA_TEMP_COUNT][CEMA_SOC_COUNT],
                       CEMA_Real soc,
                       CEMA_Real temperature_c)
{
  uint32_t index;
  if (temperature_c <= CEMA_TEMPERATURES[0])
  {
    return grid_row(grid[0], soc);
  }
  if (temperature_c >= CEMA_TEMPERATURES[CEMA_TEMP_COUNT - 1U])
  {
    return grid_row(grid[CEMA_TEMP_COUNT - 1U], soc);
  }
  for (index = 1U; index < CEMA_TEMP_COUNT; ++index)
  {
    if (temperature_c <= CEMA_TEMPERATURES[index])
    {
      const CEMA_Real x0 = CEMA_TEMPERATURES[index - 1U];
      const CEMA_Real x1 = CEMA_TEMPERATURES[index];
      const CEMA_Real weight = (temperature_c - x0) / (x1 - x0);
      const CEMA_Real y0 = grid_row(grid[index - 1U], soc);
      const CEMA_Real y1 = grid_row(grid[index], soc);
      return y0 + weight * (y1 - y0);
    }
  }
  return grid_row(grid[CEMA_TEMP_COUNT - 1U], soc);
}

static void symmetrize(CEMA_Real matrix[CEMA_MAX_STATE][CEMA_MAX_STATE],
                       uint32_t n)
{
  uint32_t i;
  uint32_t j;
  for (i = 0U; i < n; ++i)
  {
    for (j = i + 1U; j < n; ++j)
    {
      const CEMA_Real value = 0.5 * (matrix[i][j] + matrix[j][i]);
      matrix[i][j] = value;
      matrix[j][i] = value;
    }
  }
}

static void project_psd(CEMA_Real matrix[CEMA_MAX_STATE][CEMA_MAX_STATE],
                        uint32_t n,
                        CEMA_Real floor,
                        CEMA_Real ceiling)
{
  CEMA_Real a[CEMA_MAX_STATE][CEMA_MAX_STATE] = {{0.0}};
  CEMA_Real vectors[CEMA_MAX_STATE][CEMA_MAX_STATE] = {{0.0}};
  CEMA_Real rebuilt[CEMA_MAX_STATE][CEMA_MAX_STATE] = {{0.0}};
  uint32_t i;
  uint32_t j;
  uint32_t k;
  uint32_t iteration;

  symmetrize(matrix, n);
  for (i = 0U; i < n; ++i)
  {
    vectors[i][i] = 1.0F;
    for (j = 0U; j < n; ++j)
    {
      a[i][j] = matrix[i][j];
    }
  }
#if CEMA_METHOD != CEMA_METHOD_CC
  for (iteration = 0U; iteration < 64U; ++iteration)
#else
  for (iteration = 0U; iteration < 20U; ++iteration)
#endif
  {
    uint32_t p = 0U;
    uint32_t q = 1U;
    CEMA_Real largest = 0.0;
    for (i = 0U; i < n; ++i)
    {
      for (j = i + 1U; j < n; ++j)
      {
        const CEMA_Real candidate = cema_abs(a[i][j]);
        if (candidate > largest)
        {
          largest = candidate;
          p = i;
          q = j;
        }
      }
    }
#if CEMA_METHOD != CEMA_METHOD_CC
    if (largest < 1.0e-15)
#else
    if (largest < 1.0e-12F)
#endif
    {
      break;
    }
    {
      const CEMA_Real angle =
          0.5 * cema_atan2(2.0 * a[p][q], a[q][q] - a[p][p]);
      const CEMA_Real cosine = cema_cos(angle);
      const CEMA_Real sine = cema_sin(angle);
      const CEMA_Real app = a[p][p];
      const CEMA_Real aqq = a[q][q];
      const CEMA_Real apq = a[p][q];
      a[p][p] = cosine * cosine * app - 2.0F * sine * cosine * apq +
                sine * sine * aqq;
      a[q][q] = sine * sine * app + 2.0F * sine * cosine * apq +
                cosine * cosine * aqq;
      a[p][q] = 0.0F;
      a[q][p] = 0.0F;
      for (k = 0U; k < n; ++k)
      {
        if ((k != p) && (k != q))
        {
          const CEMA_Real akp = a[k][p];
          const CEMA_Real akq = a[k][q];
          a[k][p] = cosine * akp - sine * akq;
          a[p][k] = a[k][p];
          a[k][q] = sine * akp + cosine * akq;
          a[q][k] = a[k][q];
        }
        {
          const CEMA_Real vkp = vectors[k][p];
          const CEMA_Real vkq = vectors[k][q];
          vectors[k][p] = cosine * vkp - sine * vkq;
          vectors[k][q] = sine * vkp + cosine * vkq;
        }
      }
    }
  }
  for (k = 0U; k < n; ++k)
  {
    const CEMA_Real eigenvalue =
        clamp_real(a[k][k], floor, ceiling);
    for (i = 0U; i < n; ++i)
    {
      for (j = 0U; j < n; ++j)
      {
        rebuilt[i][j] +=
            vectors[i][k] * eigenvalue * vectors[j][k];
      }
    }
  }
  for (i = 0U; i < n; ++i)
  {
    for (j = 0U; j < n; ++j)
    {
      matrix[i][j] = rebuilt[i][j];
    }
  }
}

static int cholesky_scaled(
    const CEMA_Real covariance[CEMA_MAX_STATE][CEMA_MAX_STATE],
    CEMA_Real scale,
    uint32_t n,
    CEMA_Real root[CEMA_MAX_STATE][CEMA_MAX_STATE])
{
  uint32_t attempt;
  for (attempt = 0U; attempt < 8U; ++attempt)
  {
    const CEMA_Real jitter =
        cema_pow(10.0, (CEMA_Real)((int32_t)attempt - 12));
    uint32_t i;
    uint32_t j;
    int ok = 1;
    memset(root, 0, sizeof(CEMA_Real) * CEMA_MAX_STATE * CEMA_MAX_STATE);
    for (i = 0U; i < n; ++i)
    {
      for (j = 0U; j <= i; ++j)
      {
        CEMA_Real value = scale * covariance[i][j];
        uint32_t k;
        if (i == j)
        {
#if CEMA_CHEMISTRY == CEMA_CHEMISTRY_NMC
          value += scale * jitter;
#else
          value += jitter;
#endif
        }
        for (k = 0U; k < j; ++k)
        {
          value -= root[i][k] * root[j][k];
        }
        if (i == j)
        {
          if (!(value > 0.0F) || !isfinite(value))
          {
            ok = 0;
            break;
          }
          root[i][j] = cema_sqrt(value);
        }
        else
        {
          if (!(root[j][j] > 0.0F))
          {
            ok = 0;
            break;
          }
          root[i][j] = value / root[j][j];
        }
      }
      if (!ok)
      {
        break;
      }
    }
    if (ok)
    {
      return 1;
    }
  }
  return 0;
}

static void parameter_values(CEMA_Real temperature_c,
                             CEMA_Real *r0,
                             CEMA_Real *r1,
                             CEMA_Real *r2,
                             CEMA_Real *tau1,
                             CEMA_Real *tau2,
                             CEMA_Real *gamma)
{
  *r0 = interp_temperature(CEMA_R0, temperature_c);
  *r1 = interp_temperature(CEMA_R1, temperature_c);
  *r2 = interp_temperature(CEMA_R2, temperature_c);
  *tau1 = interp_temperature(CEMA_TAU1, temperature_c);
  *tau2 = interp_temperature(CEMA_TAU2, temperature_c);
#if CEMA_CHEMISTRY == CEMA_CHEMISTRY_LFP
  *gamma = g_state.gamma;
#else
  *gamma = interp_temperature(CEMA_GAMMA, temperature_c);
#endif
}

static CEMA_Real q_value(
    const CEMA_AssetReal *values, CEMA_Real temperature_c)
{
  return interp_temperature(values, temperature_c);
}

static void propagate_vector(CEMA_Real *state,
                             CEMA_Real previous_current,
                             CEMA_Real dt_s,
                             CEMA_Real q_ref,
                             CEMA_Real r1,
                             CEMA_Real r2,
                             CEMA_Real tau1,
                             CEMA_Real tau2,
                             CEMA_Real gamma)
{
  const CEMA_Real effective_dt = fmax(dt_s, 0.0);
#if CEMA_CHEMISTRY == CEMA_CHEMISTRY_LFP
  const CEMA_Real a1 =
      cema_exp(-effective_dt / fmax(tau1, effective_dt + 1.0e-9));
  const CEMA_Real a2 =
      cema_exp(-effective_dt / fmax(tau2, effective_dt + 1.0e-9));
#else
  const CEMA_Real a1 = cema_exp(-effective_dt / fmax(tau1, 1.0e-9));
  const CEMA_Real a2 = cema_exp(-effective_dt / fmax(tau2, 1.0e-9));
#endif
  state[0] = clamp_real(
      state[0] - previous_current * effective_dt / (3600.0 * q_ref),
      0.0,
      1.0);
  state[1] =
      a1 * state[1] + r1 * (1.0 - a1) * previous_current;
#if CEMA_ORDER == 2
  state[2] =
      a2 * state[2] + r2 * (1.0 - a2) * previous_current;
#endif
#if CEMA_WITH_HYSTERESIS
  {
    const CEMA_Real ah = cema_exp(
        -fmax(gamma, 0.0) * cema_abs(previous_current) * effective_dt /
        (3600.0 * fmax(q_ref, 1.0e-12)));
    if (cema_abs(previous_current) > 1.0e-12)
    {
      const CEMA_Real sign = (previous_current > 0.0) ? 1.0 : -1.0;
      state[3] =
          clamp_real(ah * state[3] + (1.0 - ah) * (-sign), -1.0, 1.0);
    }
  }
#else
  (void)gamma;
#endif
}

static CEMA_Real ocv_base(CEMA_Real soc, CEMA_Real temperature_c)
{
#if CEMA_CHEMISTRY == CEMA_CHEMISTRY_NMC
  return nmc_pchip_eval(soc, temperature_c, 0);
#else
  return grid_eval(CEMA_OCV_BASE, soc, temperature_c);
#endif
}

static CEMA_Real ocv_base_slope(CEMA_Real soc, CEMA_Real temperature_c)
{
#if CEMA_CHEMISTRY == CEMA_CHEMISTRY_NMC
  return nmc_pchip_eval(soc, temperature_c, 1);
#else
  return grid_eval(CEMA_OCV_DBASE, soc, temperature_c);
#endif
}

static CEMA_Real terminal_voltage(const CEMA_Real *state,
                                  CEMA_Real current,
                                  CEMA_Real temperature_c,
                                  CEMA_Real r0)
{
  CEMA_Real voltage =
      ocv_base(state[0], temperature_c) - current * r0 - state[1];
#if CEMA_ORDER == 2
  voltage -= state[2];
#endif
#if CEMA_WITH_HYSTERESIS
  voltage +=
      grid_eval(CEMA_OCV_HMAG, state[0], temperature_c) * state[3];
#endif
  return voltage;
}

static void observation_vector(const CEMA_Real *state,
                               CEMA_Real temperature_c,
                               uint32_t n,
                               CEMA_Real *observation)
{
  uint32_t index;
  for (index = 0U; index < n; ++index)
  {
    observation[index] = 0.0F;
  }
  observation[0] =
      clamp_real(ocv_base_slope(state[0], temperature_c), CEMA_SLOPE_MIN,
                 CEMA_SLOPE_MAX);
  observation[1] = -1.0F;
#if CEMA_ORDER == 2
  observation[2] = -1.0F;
#endif
#if CEMA_WITH_HYSTERESIS
  observation[0] +=
      clamp_real(grid_eval(CEMA_OCV_DHMAG, state[0], temperature_c),
                 -10.0,
                 10.0) *
      state[3];
  observation[3] = grid_eval(CEMA_OCV_HMAG, state[0], temperature_c);
#endif
}

static CEMA_Real measurement_variance(
    CEMA_Real soc, CEMA_Real temperature_c)
{
  CEMA_Real base = g_state.adaptive_r;
#if CEMA_CHEMISTRY == CEMA_CHEMISTRY_NMC
  (void)soc;
  (void)temperature_c;
  return base;
#else
  const CEMA_Real slope =
      cema_abs(grid_eval(CEMA_OCV_DISCHARGE_SLOPE, soc, temperature_c));
  CEMA_Real factor =
      CEMA_SLOPE_SREF /
      fmax(slope, CEMA_SLOPE_SMIN);
  factor = clamp_real(factor * factor, CEMA_SLOPE_FACTOR_MIN,
                  CEMA_SLOPE_FACTOR_MAX);
  return base * factor;
#endif
}

static CEMA_Real ekf_step(CEMA_Real voltage,
                          CEMA_Real current,
                          CEMA_Real temperature_c,
                          CEMA_Real dt_s)
{
  const uint32_t n = CEMA_STATE_DIM;
  CEMA_Real r0;
  CEMA_Real r1;
  CEMA_Real r2;
  CEMA_Real tau1;
  CEMA_Real tau2;
  CEMA_Real gamma;
  CEMA_Real observation[CEMA_MAX_STATE] = {0.0};
  CEMA_Real ph[CEMA_MAX_STATE] = {0.0};
  CEMA_Real gain[CEMA_MAX_STATE] = {0.0};
  CEMA_Real a[CEMA_MAX_STATE][CEMA_MAX_STATE] = {{0.0}};
  CEMA_Real ap[CEMA_MAX_STATE][CEMA_MAX_STATE] = {{0.0}};
  CEMA_Real updated[CEMA_MAX_STATE][CEMA_MAX_STATE] = {{0.0}};
  const CEMA_Real q_ref = g_state.q_ref_ah;
  CEMA_Real predicted;
  CEMA_Real residual;
  CEMA_Real innovation_variance;
  CEMA_Real r_used;
  uint32_t i;
  uint32_t j;
  uint32_t k;

  parameter_values(
      temperature_c, &r0, &r1, &r2, &tau1, &tau2, &gamma);
  if (g_state.sample_count > 0U)
  {
    CEMA_Real fdiag[CEMA_MAX_STATE] = {1.0, 1.0, 1.0, 1.0};
    const CEMA_Real effective_dt = fmax(dt_s, 0.0);
#if CEMA_CHEMISTRY == CEMA_CHEMISTRY_LFP
    fdiag[1] =
        cema_exp(-effective_dt / fmax(tau1, effective_dt + 1.0e-9));
    fdiag[2] =
        cema_exp(-effective_dt / fmax(tau2, effective_dt + 1.0e-9));
#else
    fdiag[1] = cema_exp(-effective_dt / fmax(tau1, 1.0e-9));
    fdiag[2] = cema_exp(-effective_dt / fmax(tau2, 1.0e-9));
#endif
#if CEMA_WITH_HYSTERESIS
    fdiag[3] = cema_exp(
        -fmax(gamma, 0.0) * cema_abs(g_state.previous_current_a) *
        effective_dt / (3600.0 * fmax(q_ref, 1.0e-12)));
#endif
    propagate_vector(
        g_state.x, g_state.previous_current_a, effective_dt, q_ref, r1, r2,
        tau1, tau2, gamma);
    for (i = 0U; i < n; ++i)
    {
      for (j = 0U; j < n; ++j)
      {
        g_state.p[i][j] *= fdiag[i] * fdiag[j];
      }
#if CEMA_CHEMISTRY == CEMA_CHEMISTRY_NMC
      g_state.p[i][i] +=
          q_value((i == 0U) ? CEMA_Q_SOC : CEMA_Q_VP, temperature_c) *
          effective_dt;
#else
      if (i == 0U)
      {
        g_state.p[i][i] += q_value(CEMA_Q_SOC, temperature_c);
      }
      else if (i == 3U)
      {
        g_state.p[i][i] += q_value(CEMA_Q_H, temperature_c);
      }
      else
      {
        g_state.p[i][i] += q_value(CEMA_Q_VP, temperature_c);
      }
#endif
    }
  }

  predicted = terminal_voltage(g_state.x, current, temperature_c, r0);
  residual = voltage - predicted;
#if CEMA_CHEMISTRY == CEMA_CHEMISTRY_NMC && CEMA_ADAPTIVE
  {
    const CEMA_Real beta =
        interp_temperature(CEMA_ADAPTIVE_BETA, temperature_c);
    const CEMA_Real base =
        interp_temperature(CEMA_R_VOLTAGE, temperature_c);
    const CEMA_Real candidate =
        (1.0F - beta) * g_state.adaptive_r + beta * residual * residual;
    g_state.adaptive_r =
        clamp_real(candidate, base * CEMA_ADAPTIVE_R_MIN_SCALE,
               base * CEMA_ADAPTIVE_R_MAX_SCALE);
  }
#endif
  observation_vector(g_state.x, temperature_c, n, observation);
  for (i = 0U; i < n; ++i)
  {
    for (j = 0U; j < n; ++j)
    {
      ph[i] += g_state.p[i][j] * observation[j];
    }
  }
  r_used = measurement_variance(g_state.x[0], temperature_c);
  innovation_variance = r_used;
  for (i = 0U; i < n; ++i)
  {
    innovation_variance += observation[i] * ph[i];
  }
  innovation_variance =
      fmax(innovation_variance, CEMA_INNOVATION_FLOOR);
  for (i = 0U; i < n; ++i)
  {
    gain[i] = ph[i] / innovation_variance;
    g_state.x[i] += gain[i] * residual;
  }
  g_state.x[0] = clamp_real(g_state.x[0], 0.0, 1.0);
#if CEMA_CHEMISTRY == CEMA_CHEMISTRY_NMC
  for (i = 1U; i < n; ++i)
  {
    g_state.x[i] = clamp_real(
        g_state.x[i], -CEMA_VOLTAGE_STATE_LIMIT,
        CEMA_VOLTAGE_STATE_LIMIT);
  }
#elif CEMA_WITH_HYSTERESIS
  g_state.x[3] = clamp_real(g_state.x[3], -1.0, 1.0);
#endif
  for (i = 0U; i < n; ++i)
  {
    a[i][i] = 1.0F;
    for (j = 0U; j < n; ++j)
    {
      a[i][j] -= gain[i] * observation[j];
    }
  }
  for (i = 0U; i < n; ++i)
  {
    for (j = 0U; j < n; ++j)
    {
      for (k = 0U; k < n; ++k)
      {
        ap[i][j] += a[i][k] * g_state.p[k][j];
      }
    }
  }
  for (i = 0U; i < n; ++i)
  {
    for (j = 0U; j < n; ++j)
    {
      for (k = 0U; k < n; ++k)
      {
        updated[i][j] += ap[i][k] * a[j][k];
      }
      updated[i][j] += gain[i] * gain[j] * r_used;
    }
  }
  for (i = 0U; i < n; ++i)
  {
    for (j = 0U; j < n; ++j)
    {
      g_state.p[i][j] = updated[i][j];
    }
  }
  symmetrize(g_state.p, n);
#if CEMA_CHEMISTRY == CEMA_CHEMISTRY_NMC
  project_psd(
      g_state.p, n, CEMA_COVARIANCE_FLOOR, CEMA_REAL_MAX);
#else
  for (i = 0U; i < n; ++i)
  {
    g_state.p[i][i] = clamp_real(g_state.p[i][i], 1.0e-12, 1.0);
  }
#if CEMA_ADAPTIVE
  {
    CEMA_Real predicted_variance = 0.0;
    CEMA_Real target;
    for (i = 0U; i < n; ++i)
    {
      for (j = 0U; j < n; ++j)
      {
        predicted_variance +=
            observation[i] * g_state.p[i][j] * observation[j];
      }
    }
    target =
        fmax(residual * residual - predicted_variance, CEMA_ADAPTIVE_R_MIN);
    g_state.adaptive_r = clamp_real(
        (1.0F - CEMA_ADAPTIVE_ALPHA) * g_state.adaptive_r +
            CEMA_ADAPTIVE_ALPHA * target,
        CEMA_ADAPTIVE_R_MIN,
        CEMA_ADAPTIVE_R_MAX);
  }
#endif
#endif
  return g_state.x[0];
}

#if CEMA_UKF
static CEMA_Real ukf_step(CEMA_Real voltage,
                          CEMA_Real current,
                          CEMA_Real temperature_c,
                          CEMA_Real dt_s)
{
  const uint32_t n = CEMA_STATE_DIM;
  const uint32_t point_count = 2U * CEMA_STATE_DIM + 1U;
  const CEMA_Real alpha = 0.1;
  const CEMA_Real beta = 2.0;
  const CEMA_Real lambda = alpha * alpha * (CEMA_Real)n - (CEMA_Real)n;
  const CEMA_Real scale = (CEMA_Real)n + lambda;
  const CEMA_Real weight = 1.0 / (2.0 * scale);
  const CEMA_Real weight0_mean = lambda / scale;
  const CEMA_Real weight0_cov =
      weight0_mean + 1.0 - alpha * alpha + beta;
  CEMA_Real sigma[CEMA_MAX_SIGMA][CEMA_MAX_STATE] = {{0.0}};
  CEMA_Real root[CEMA_MAX_STATE][CEMA_MAX_STATE] = {{0.0}};
  CEMA_Real zsig[CEMA_MAX_SIGMA] = {0.0};
  CEMA_Real mean[CEMA_MAX_STATE] = {0.0};
  CEMA_Real covariance[CEMA_MAX_STATE][CEMA_MAX_STATE] = {{0.0}};
  CEMA_Real cross[CEMA_MAX_STATE] = {0.0};
  CEMA_Real gain[CEMA_MAX_STATE] = {0.0};
  CEMA_Real r0;
  CEMA_Real r1;
  CEMA_Real r2;
  CEMA_Real tau1;
  CEMA_Real tau2;
  CEMA_Real gamma;
  const CEMA_Real q_ref = g_state.q_ref_ah;
  CEMA_Real zmean = 0.0;
  CEMA_Real innovation_variance;
  CEMA_Real residual;
  uint32_t i;
  uint32_t j;
  uint32_t q;

  parameter_values(
      temperature_c, &r0, &r1, &r2, &tau1, &tau2, &gamma);
#if CEMA_CHEMISTRY == CEMA_CHEMISTRY_LFP
  project_psd(g_state.p, n, 1.0e-12F, 1.0F);
#endif
  if (!cholesky_scaled(g_state.p, scale, n, root))
  {
    g_state.status = CEMA_STATUS_NUMERIC;
    return NAN;
  }
  for (i = 0U; i < n; ++i)
  {
    sigma[0][i] = g_state.x[i];
  }
  for (j = 0U; j < n; ++j)
  {
    for (i = 0U; i < n; ++i)
    {
      sigma[1U + j][i] = g_state.x[i] + root[i][j];
      sigma[1U + n + j][i] = g_state.x[i] - root[i][j];
    }
  }
  if (g_state.sample_count > 0U)
  {
    for (q = 0U; q < point_count; ++q)
    {
      propagate_vector(
          sigma[q], g_state.previous_current_a, dt_s, q_ref, r1, r2, tau1,
          tau2, gamma);
    }
    for (i = 0U; i < n; ++i)
    {
      mean[i] = weight0_mean * sigma[0][i];
      for (q = 1U; q < point_count; ++q)
      {
        mean[i] += weight * sigma[q][i];
      }
    }
#if CEMA_CHEMISTRY == CEMA_CHEMISTRY_LFP
    mean[0] = clamp_real(mean[0], 0.0, 1.0);
    mean[3] = clamp_real(mean[3], -1.0, 1.0);
#endif
    for (q = 0U; q < point_count; ++q)
    {
      const CEMA_Real w = (q == 0U) ? weight0_cov : weight;
      for (i = 0U; i < n; ++i)
      {
        const CEMA_Real di = sigma[q][i] - mean[i];
        for (j = 0U; j < n; ++j)
        {
          covariance[i][j] +=
              w * di * (sigma[q][j] - mean[j]);
        }
      }
    }
    for (i = 0U; i < n; ++i)
    {
#if CEMA_CHEMISTRY == CEMA_CHEMISTRY_NMC
      covariance[i][i] +=
          q_value((i == 0U) ? CEMA_Q_SOC : CEMA_Q_VP, temperature_c) *
          fmax(dt_s, 0.0);
#else
      covariance[i][i] += q_value(
          (i == 0U) ? CEMA_Q_SOC : ((i == 3U) ? CEMA_Q_H : CEMA_Q_VP),
          temperature_c);
#endif
    }
    memcpy(g_state.x, mean, sizeof(CEMA_Real) * n);
    memcpy(g_state.p, covariance, sizeof(covariance));
#if CEMA_CHEMISTRY == CEMA_CHEMISTRY_LFP
    project_psd(g_state.p, n, 1.0e-12F, 1.0F);
#endif
    if (!cholesky_scaled(g_state.p, scale, n, root))
    {
      g_state.status = CEMA_STATUS_NUMERIC;
      return NAN;
    }
    for (i = 0U; i < n; ++i)
    {
      sigma[0][i] = g_state.x[i];
    }
    for (j = 0U; j < n; ++j)
    {
      for (i = 0U; i < n; ++i)
      {
        sigma[1U + j][i] = g_state.x[i] + root[i][j];
        sigma[1U + n + j][i] = g_state.x[i] - root[i][j];
      }
    }
  }
  for (q = 0U; q < point_count; ++q)
  {
    zsig[q] = terminal_voltage(sigma[q], current, temperature_c, r0);
  }
  zmean = weight0_mean * zsig[0];
  for (q = 1U; q < point_count; ++q)
  {
    zmean += weight * zsig[q];
  }
  innovation_variance =
      measurement_variance(g_state.x[0], temperature_c);
  for (q = 0U; q < point_count; ++q)
  {
    const CEMA_Real w = (q == 0U) ? weight0_cov : weight;
    const CEMA_Real dz = zsig[q] - zmean;
    innovation_variance += w * dz * dz;
    for (i = 0U; i < n; ++i)
    {
      cross[i] += w * (sigma[q][i] - g_state.x[i]) * dz;
    }
  }
  innovation_variance =
      fmax(innovation_variance, CEMA_INNOVATION_FLOOR);
  residual = voltage - zmean;
  for (i = 0U; i < n; ++i)
  {
    gain[i] = cross[i] / innovation_variance;
    g_state.x[i] += gain[i] * residual;
  }
  g_state.x[0] = clamp_real(g_state.x[0], 0.0, 1.0);
#if CEMA_WITH_HYSTERESIS
  g_state.x[3] = clamp_real(g_state.x[3], -1.0, 1.0);
#else
  for (i = 1U; i < n; ++i)
  {
    g_state.x[i] = clamp_real(
        g_state.x[i], -CEMA_VOLTAGE_STATE_LIMIT,
        CEMA_VOLTAGE_STATE_LIMIT);
  }
#endif
  for (i = 0U; i < n; ++i)
  {
    for (j = 0U; j < n; ++j)
    {
      g_state.p[i][j] -=
          gain[i] * gain[j] * innovation_variance;
    }
  }
  symmetrize(g_state.p, n);
#if CEMA_CHEMISTRY == CEMA_CHEMISTRY_NMC
  project_psd(
      g_state.p, n, CEMA_COVARIANCE_FLOOR, CEMA_REAL_MAX);
#else
  project_psd(g_state.p, n, 1.0e-12F, 1.0F);
#endif
  return g_state.x[0];
}
#endif
#endif

void CEMA_KF_Reset_State(const CEMA_KF_Reset *reset)
{
  uint32_t i;
  memset(&g_state, 0, sizeof(g_state));
  g_state.status = CEMA_STATUS_OK;
  g_state.x[0] = clamp_real(reset->initial_soc, 0.0, 1.0);
  g_state.q_ref_ah =
      nearest_temperature(CEMA_QREF, reset->nominal_temperature_c);
#if CEMA_METHOD != CEMA_METHOD_CC
#if CEMA_CHEMISTRY == CEMA_CHEMISTRY_LFP
  g_state.gamma =
      nearest_temperature(CEMA_GAMMA, reset->nominal_temperature_c);
#else
  g_state.gamma =
      interp_temperature(CEMA_GAMMA, reset->nominal_temperature_c);
#endif
#if CEMA_CHEMISTRY == CEMA_CHEMISTRY_NMC
  for (i = 0U; i < CEMA_STATE_DIM; ++i)
  {
    g_state.p[i][i] = 0.01F;
  }
#else
  g_state.p[0][0] = 2.5e-3F;
  g_state.p[1][1] = 2.5e-3F;
  g_state.p[2][2] = 2.5e-3F;
#if CEMA_WITH_HYSTERESIS
  g_state.p[3][3] = 0.25F;
  {
    const CEMA_Real mid =
        grid_eval(CEMA_OCV_RAW_MID, g_state.x[0],
                  reset->initial_temperature_c);
    const CEMA_Real magnitude =
        grid_eval(CEMA_OCV_HMAG, g_state.x[0],
                  reset->initial_temperature_c);
    g_state.x[3] =
        (magnitude <= 1.0e-12F)
            ? 0.0F
            : clamp_real((reset->initial_voltage_v - mid) / magnitude,
                         -1.0, 1.0);
  }
#endif
#endif
  g_state.adaptive_r =
      interp_temperature(CEMA_R_VOLTAGE, reset->initial_temperature_c);
#else
  (void)i;
#endif
}

double CEMA_KF_Step(double voltage_v,
                    double raw_current_a,
                    double temperature_c,
                    double dt_s)
{
  const CEMA_Real voltage = (CEMA_Real)voltage_v;
  const CEMA_Real current = (CEMA_Real)(-raw_current_a);
  const CEMA_Real temperature = (CEMA_Real)temperature_c;
  const CEMA_Real dt = (CEMA_Real)dt_s;
  CEMA_Real output;
  if (g_state.status != CEMA_STATUS_OK)
  {
    return NAN;
  }
#if CEMA_METHOD == CEMA_METHOD_CC
  if (g_state.sample_count > 0U)
  {
    const CEMA_Real q_ref = g_state.q_ref_ah;
    g_state.x[0] = clamp_real(
        g_state.x[0] -
            g_state.previous_current_a * fmax(dt, 0.0) /
                (3600.0 * q_ref),
        0.0,
        1.0);
  }
  output = g_state.x[0];
#elif CEMA_UKF
  output = ukf_step(voltage, current, temperature, dt);
#else
  output = ekf_step(voltage, current, temperature, dt);
#endif
  g_state.previous_current_a = current;
  ++g_state.sample_count;
  if (!isfinite(output))
  {
    g_state.status = CEMA_STATUS_NUMERIC;
  }
  return (double)output;
}

uint32_t CEMA_KF_Status(void) { return g_state.status; }
uint32_t CEMA_KF_Chemistry(void) { return CEMA_CHEMISTRY; }
uint32_t CEMA_KF_Method(void) { return CEMA_METHOD; }
uint32_t CEMA_KF_State_Dim(void) { return CEMA_STATE_DIM; }
uint32_t CEMA_KF_Asset_Bytes(void) { return CEMA_ASSET_BYTES; }
uint32_t CEMA_KF_Runtime_State_Bytes(void)
{
  return (uint32_t)sizeof(g_state);
}
uint32_t CEMA_KF_Sample_Count(void) { return g_state.sample_count; }
const char *CEMA_KF_Model_Id(void) { return CEMA_MODEL_ID; }
