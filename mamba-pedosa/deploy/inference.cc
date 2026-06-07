#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "model_data.h"

// 6MB Tensor Arena for Cortex-M55 SRAM
constexpr int kTensorArenaSize = 6 * 1024 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input_ecg = nullptr;
  TfLiteTensor* input_spo2 = nullptr;
  TfLiteTensor* input_ptt = nullptr;
  TfLiteTensor* input_clinical = nullptr;
  TfLiteTensor* output_gamma = nullptr;
  TfLiteTensor* output_nu = nullptr;
  TfLiteTensor* output_alpha = nullptr;
  TfLiteTensor* output_beta = nullptr;
}

void setup() {
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema version mismatch!");
    return;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Get pointers to input and output tensors
  input_ecg = interpreter->input(0);
  input_spo2 = interpreter->input(1);
  input_ptt = interpreter->input(2);
  input_clinical = interpreter->input(3);
  
  // Note: Output indices may vary depending on ONNX-TF mapping.
  output_gamma = interpreter->output(0);
  output_nu = interpreter->output(1);
  output_alpha = interpreter->output(2);
  output_beta = interpreter->output(3);
}

void loop() {
  // 1. Fetch sensor data into inputs (dummy for now)
  // Inputs are INT8 due to quantization
  for (int i = 0; i < 7680; ++i) {
    input_ecg->data.int8[i] = 0; // Populate with actual sensor reading
  }

  // 2. Invoke interpreter
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    return;
  }

  // 3. Read outputs and dequantize
  // Dequantization: float_val = (int8_val - zero_point) * scale
  float gamma = (output_gamma->data.int8[0] - output_gamma->params.zero_point) * output_gamma->params.scale;
  float nu = (output_nu->data.int8[0] - output_nu->params.zero_point) * output_nu->params.scale;
  float alpha = (output_alpha->data.int8[0] - output_alpha->params.zero_point) * output_alpha->params.scale;
  
  // 4. Calculate Evidence Score S
  float evidence_score = nu + 2.0f * alpha;
  
  error_reporter->Report("Predicted AHI: %f | Confidence: %f", gamma, evidence_score);
  
  // Wait before next epoch (e.g. 30 seconds)
}
