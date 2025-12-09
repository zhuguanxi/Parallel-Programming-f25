#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  __pp_vec_float x, result, clampVal;
  __pp_vec_int exp, one, zero;
  __pp_mask maskAll, maskExpNotZero, maskClamp;

  one = _pp_vset_int(1);
  zero = _pp_vset_int(0);
  clampVal = _pp_vset_float(9.999999f);

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    int width = (i + VECTOR_WIDTH <= N) ? VECTOR_WIDTH : (N - i);
    maskAll = _pp_init_ones(width);

    _pp_vload_float(x, values + i, maskAll);
    _pp_vload_int(exp, exponents + i, maskAll);

    _pp_vset_float(result, 1.0f, maskAll);

    _pp_vgt_int(maskExpNotZero, exp, zero, maskAll);

    while (_pp_cntbits(maskExpNotZero) > 0)
    {
      _pp_vmult_float(result, result, x, maskExpNotZero);   
      _pp_vsub_int(exp, exp, one, maskExpNotZero);         
      _pp_vgt_int(maskExpNotZero, exp, zero, maskAll);      
    }

    _pp_vgt_float(maskClamp, result, clampVal, maskAll);
    _pp_vmove_float(result, clampVal, maskClamp);

    _pp_vstore_float(output + i, result, maskAll);
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{
  __pp_vec_float x;
  __pp_mask maskAll;

  float total = 0.f;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    maskAll = _pp_init_ones();
    _pp_vload_float(x, values + i, maskAll);

    for (int step = VECTOR_WIDTH / 2; step > 0; step /= 2)
    {
      __pp_vec_float tmp;
      _pp_hadd_float(tmp, x);
      _pp_interleave_float(x, tmp);
    }

    // 取出第一個元素
    float partial;
    __pp_mask maskOne = _pp_init_ones(1);
    _pp_vstore_float(&partial, x, maskOne);
    total += partial;
  }

  return total;
}

