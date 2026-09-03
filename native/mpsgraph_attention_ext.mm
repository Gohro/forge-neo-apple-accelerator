#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSGeneratorImpl.h>
#include <ATen/native/mps/OperationUtils.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <map>
#include <mutex>
#include <sstream>
#include <string>

namespace py = pybind11;

namespace {

constexpr int64_t kBatch = 2;
constexpr int64_t kHeads = 16;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kFlatDim = kHeads * kHeadDim;

std::mutex g_graph_mutex;
enum class AttentionVariant {
  SDPA,
  SoftmaxAPI,
  ManualSoftmax,
};

struct AttentionGraphCache {
  MPSGraph* graph = nil;
  MPSGraphTensor* q = nil;
  MPSGraphTensor* k = nil;
  MPSGraphTensor* v = nil;
  MPSGraphTensor* out = nil;
  int64_t batch = 0;
  int64_t q_sequence = 0;
  int64_t kv_sequence = 0;
  int64_t heads = 0;
  int64_t head_dim = 0;
};

std::map<std::string, AttentionGraphCache> g_attention_caches;

struct LinearGraphCache {
  MPSGraph* graph = nil;
  MPSGraphTensor* input = nil;
  MPSGraphTensor* weight = nil;
  MPSGraphTensor* out = nil;
  int64_t batch = 0;
  int64_t in_features = 0;
  int64_t out_features = 0;
};

std::map<std::string, LinearGraphCache> g_linear_caches;

struct LegacyGeluGraphCache {
  MPSGraph* graph = nil;
  MPSGraphTensor* input = nil;
  MPSGraphTensor* out = nil;
};

std::map<std::string, LegacyGeluGraphCache> g_legacy_gelu_caches;

struct LegacyNormalGraphCache {
  MPSGraph* graph = nil;
  MPSGraphTensor* state = nil;
  MPSGraphTensor* out = nil;
};

std::map<std::string, LegacyNormalGraphCache> g_legacy_normal_caches;

struct FusedSelfAttentionGraphCache {
  MPSGraph* graph = nil;
  MPSGraphTensor* input = nil;
  MPSGraphTensor* rope = nil;
  MPSGraphTensor* q_weight = nil;
  MPSGraphTensor* k_weight = nil;
  MPSGraphTensor* v_weight = nil;
  MPSGraphTensor* output_weight = nil;
  MPSGraphTensor* q_norm_weight = nil;
  MPSGraphTensor* k_norm_weight = nil;
  MPSGraphTensor* out = nil;
  int64_t batch = 0;
  int64_t sequence = 0;
};

std::map<std::string, FusedSelfAttentionGraphCache> g_fused_self_attention_caches;

std::string shape_string(const at::Tensor& tensor) {
  std::ostringstream stream;
  stream << "[";
  for (int64_t i = 0; i < tensor.dim(); ++i) {
    if (i) {
      stream << ", ";
    }
    stream << tensor.size(i);
  }
  stream << "]";
  return stream.str();
}

void check_attention_tensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_mps(), name, " must be an MPS tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be torch.bfloat16");
  TORCH_CHECK(tensor.dim() == 4, name, " must be rank 4 B,S,H,D; got ", shape_string(tensor));
  TORCH_CHECK(tensor.size(0) == kBatch && tensor.size(2) == kHeads &&
                  tensor.size(3) == kHeadDim,
              name,
              " must have shape [2, S, 16, 128]; got ",
              shape_string(tensor));
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.storage_offset() == 0, name, " must have storage_offset=0");
}

void check_linear_tensor(const at::Tensor& input, const at::Tensor& weight) {
  TORCH_CHECK(input.is_mps(), "linear input must be an MPS tensor");
  TORCH_CHECK(weight.is_mps(), "linear weight must be an MPS tensor");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "linear input must be torch.bfloat16");
  TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "linear weight must be torch.bfloat16");
  TORCH_CHECK(input.dim() == 2, "linear input must be rank 2 [N, K]; got ", shape_string(input));
  TORCH_CHECK(weight.dim() == 2, "linear weight must be rank 2 [O, K]; got ", shape_string(weight));
  TORCH_CHECK(input.size(1) == weight.size(1), "linear K mismatch: input ", shape_string(input), " weight ", shape_string(weight));
  TORCH_CHECK(input.is_contiguous(), "linear input must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "linear weight must be contiguous");
  TORCH_CHECK(input.storage_offset() == 0, "linear input must have storage_offset=0");
  TORCH_CHECK(weight.storage_offset() == 0, "linear weight must have storage_offset=0");
}

void check_fused_self_attention_tensors(
    const at::Tensor& input,
    const at::Tensor& rope,
    const at::Tensor& q_weight,
    const at::Tensor& k_weight,
    const at::Tensor& v_weight,
    const at::Tensor& output_weight,
    const at::Tensor& q_norm_weight,
    const at::Tensor& k_norm_weight) {
  TORCH_CHECK(input.is_mps(), "fused self-attention input must be an MPS tensor");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "fused self-attention input must be torch.bfloat16");
  TORCH_CHECK(input.dim() == 3 && input.size(0) == kBatch && input.size(2) == kFlatDim,
              "fused self-attention input must have shape [2, S, 2048]; got ", shape_string(input));
  TORCH_CHECK(input.is_contiguous() && input.storage_offset() == 0,
              "fused self-attention input must be contiguous with storage_offset=0");
  const at::Tensor weights[] = {q_weight, k_weight, v_weight, output_weight};
  const char* weight_names[] = {"q_weight", "k_weight", "v_weight", "output_weight"};
  for (int index = 0; index < 4; ++index) {
    const at::Tensor& weight = weights[index];
    TORCH_CHECK(weight.is_mps(), weight_names[index], " must be an MPS tensor");
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16, weight_names[index], " must be torch.bfloat16");
    TORCH_CHECK(weight.dim() == 2 && weight.size(0) == kFlatDim && weight.size(1) == kFlatDim,
                weight_names[index], " must have shape [2048, 2048]; got ", shape_string(weight));
    TORCH_CHECK(weight.is_contiguous() && weight.storage_offset() == 0,
                weight_names[index], " must be contiguous with storage_offset=0");
  }
  const at::Tensor norm_weights[] = {q_norm_weight, k_norm_weight};
  const char* norm_names[] = {"q_norm_weight", "k_norm_weight"};
  for (int index = 0; index < 2; ++index) {
    const at::Tensor& weight = norm_weights[index];
    TORCH_CHECK(weight.is_mps(), norm_names[index], " must be an MPS tensor");
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16, norm_names[index], " must be torch.bfloat16");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == kHeadDim,
                norm_names[index], " must have shape [128]; got ", shape_string(weight));
    TORCH_CHECK(weight.is_contiguous() && weight.storage_offset() == 0,
                norm_names[index], " must be contiguous with storage_offset=0");
  }
  TORCH_CHECK(rope.is_mps(), "rope must be an MPS tensor");
  TORCH_CHECK(rope.scalar_type() == at::kFloat, "rope must be torch.float32");
  TORCH_CHECK(rope.dim() == 6 && rope.size(0) == 1 && rope.size(1) == input.size(1) &&
                  rope.size(2) == 1 && rope.size(3) == kHeadDim / 2 && rope.size(4) == 2 && rope.size(5) == 2,
              "rope must have shape [1, S, 1, 64, 2, 2]; got ", shape_string(rope));
  TORCH_CHECK(rope.is_contiguous() && rope.storage_offset() == 0,
              "rope must be contiguous with storage_offset=0");
}

void check_attention_shapes(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
  TORCH_CHECK(k.sizes() == v.sizes(), "k/v shape mismatch: ", shape_string(k), " vs ", shape_string(v));
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(2) == k.size(2) && q.size(3) == k.size(3),
              "q/k batch, head, or head-dim mismatch: ",
              shape_string(q),
              " vs ",
              shape_string(k));
}

MPSShape* bshd_shape(int64_t batch, int64_t sequence, int64_t heads, int64_t head_dim) {
  return @[
    @(batch),
    @(sequence),
    @(heads),
    @(head_dim),
  ];
}

MPSShape* output_graph_shape(int64_t batch, int64_t sequence, int64_t heads, int64_t head_dim) {
  return @[
    @(batch),
    @(sequence),
    @(heads),
    @(head_dim),
  ];
}

MPSShape* shape2(int64_t rows, int64_t cols) {
  return @[
    @(rows),
    @(cols),
  ];
}

MPSShape* shape1(int64_t size) {
  return @[@(size)];
}

MPSShape* shape3(int64_t dim0, int64_t dim1, int64_t dim2) {
  return @[@(dim0), @(dim1), @(dim2)];
}

MPSShape* tensor_shape(const at::Tensor& tensor) {
  NSMutableArray<NSNumber*>* shape = [NSMutableArray arrayWithCapacity:tensor.dim()];
  for (int64_t dimension = 0; dimension < tensor.dim(); ++dimension) {
    [shape addObject:@(tensor.size(dimension))];
  }
  return shape;
}

MPSShape* rope_shape(int64_t sequence) {
  return @[@1, @(sequence), @1, @(kHeadDim / 2), @2, @2];
}

std::string linear_cache_key(int64_t batch, int64_t in_features, int64_t out_features) {
  std::ostringstream stream;
  stream << batch << "x" << in_features << "x" << out_features;
  return stream.str();
}

std::string variant_name(AttentionVariant variant) {
  switch (variant) {
    case AttentionVariant::SDPA:
      return "sdpa";
    case AttentionVariant::SoftmaxAPI:
      return "softmax_api";
    case AttentionVariant::ManualSoftmax:
      return "manual_softmax";
  }
  return "unknown";
}

std::string attention_cache_key(AttentionVariant variant, int64_t batch, int64_t q_sequence, int64_t kv_sequence, int64_t heads, int64_t head_dim) {
  std::ostringstream stream;
  stream << variant_name(variant) << ":" << batch << "x" << q_sequence << "x" << kv_sequence << "x" << heads << "x" << head_dim;
  return stream.str();
}

AttentionVariant parse_variant(const std::string& variant) {
  if (variant == "mpsgraph-sdpa" || variant == "sdpa") {
    return AttentionVariant::SDPA;
  }
  if (variant == "mpsgraph-softmax-api" || variant == "softmax-api") {
    return AttentionVariant::SoftmaxAPI;
  }
  if (variant == "mpsgraph-manual-softmax" || variant == "manual-softmax") {
    return AttentionVariant::ManualSoftmax;
  }
  TORCH_CHECK(false, "Unsupported MPSGraph attention variant: ", variant);
}

MPSGraphTensor* scaled_scores(MPSGraph* graph, MPSGraphTensor* q_bhsd, MPSGraphTensor* k_bhsd) {
  MPSGraphTensor* q_fp32 = [graph castTensor:q_bhsd toType:MPSDataTypeFloat32 name:@"q_fp32"];
  MPSGraphTensor* k_fp32 = [graph castTensor:k_bhsd toType:MPSDataTypeFloat32 name:@"k_fp32"];
  MPSGraphTensor* k_bhds = [graph transposeTensor:k_fp32 dimension:2 withDimension:3 name:@"k_bhds"];
  MPSGraphTensor* scores =
      [graph matrixMultiplicationWithPrimaryTensor:q_fp32 secondaryTensor:k_bhds name:@"qk_scores"];
  MPSGraphTensor* scale = [graph constantWithScalar:static_cast<double>(1.0 / std::sqrt(static_cast<double>(kHeadDim)))
                                           dataType:MPSDataTypeFloat32];
  return [graph multiplicationWithPrimaryTensor:scores secondaryTensor:scale name:@"scaled_scores"];
}

MPSGraphTensor* manual_softmax(MPSGraph* graph, MPSGraphTensor* scores) {
  MPSGraphTensor* row_max = [graph reductionMaximumWithTensor:scores axis:-1 name:@"row_max"];
  MPSGraphTensor* shifted = [graph subtractionWithPrimaryTensor:scores secondaryTensor:row_max name:@"shifted_scores"];
  MPSGraphTensor* exps = [graph exponentWithTensor:shifted name:@"exp_scores"];
  MPSGraphTensor* row_sum = [graph reductionSumWithTensor:exps axis:-1 name:@"row_sum"];
  return [graph divisionWithPrimaryTensor:exps secondaryTensor:row_sum name:@"manual_softmax"];
}

MPSGraphTensor* build_attention_output(MPSGraph* graph, MPSGraphTensor* q_bhsd, MPSGraphTensor* k_bhsd, MPSGraphTensor* v_bhsd, AttentionVariant variant) {
  if (variant == AttentionVariant::SDPA) {
    return [graph scaledDotProductAttentionWithQueryTensor:q_bhsd
                                                 keyTensor:k_bhsd
                                               valueTensor:v_bhsd
                                                     scale:static_cast<float>(1.0 / std::sqrt(static_cast<double>(kHeadDim)))
                                                      name:@"sdpa"];
  }

  MPSGraphTensor* scores = scaled_scores(graph, q_bhsd, k_bhsd);
  MPSGraphTensor* weights = nil;
  if (variant == AttentionVariant::SoftmaxAPI) {
    weights = [graph softMaxWithTensor:scores axis:-1 name:@"softmax_api"];
  } else {
    weights = manual_softmax(graph, scores);
  }

  MPSGraphTensor* weights_bf16 = [graph castTensor:weights toType:MPSDataTypeBFloat16 name:@"weights_bf16"];
  return [graph matrixMultiplicationWithPrimaryTensor:weights_bf16 secondaryTensor:v_bhsd name:@"attn_value"];
}

void ensure_graph(AttentionVariant variant, AttentionGraphCache& cache, int64_t batch, int64_t q_sequence, int64_t kv_sequence, int64_t heads, int64_t head_dim) {
  if (cache.graph != nil) {
    return;
  }

  cache.batch = batch;
  cache.q_sequence = q_sequence;
  cache.kv_sequence = kv_sequence;
  cache.heads = heads;
  cache.head_dim = head_dim;
  cache.graph = [MPSGraph new];
  MPSShape* q_shape = bshd_shape(batch, q_sequence, heads, head_dim);
  MPSShape* kv_shape = bshd_shape(batch, kv_sequence, heads, head_dim);
  cache.q = [cache.graph placeholderWithShape:q_shape dataType:MPSDataTypeBFloat16 name:@"q_bshd"];
  cache.k = [cache.graph placeholderWithShape:kv_shape dataType:MPSDataTypeBFloat16 name:@"k_bshd"];
  cache.v = [cache.graph placeholderWithShape:kv_shape dataType:MPSDataTypeBFloat16 name:@"v_bshd"];

  MPSGraphTensor* q_bhsd = [cache.graph transposeTensor:cache.q dimension:1 withDimension:2 name:@"q_bhsd"];
  MPSGraphTensor* k_bhsd = [cache.graph transposeTensor:cache.k dimension:1 withDimension:2 name:@"k_bhsd"];
  MPSGraphTensor* v_bhsd = [cache.graph transposeTensor:cache.v dimension:1 withDimension:2 name:@"v_bhsd"];
  MPSGraphTensor* out_bhsd = build_attention_output(cache.graph, q_bhsd, k_bhsd, v_bhsd, variant);
  cache.out = [cache.graph transposeTensor:out_bhsd dimension:1 withDimension:2 name:@"out_bshd"];
}

void ensure_linear_graph(int64_t batch, int64_t in_features, int64_t out_features, LinearGraphCache& cache) {
  if (cache.graph != nil) {
    return;
  }

  cache.batch = batch;
  cache.in_features = in_features;
  cache.out_features = out_features;
  cache.graph = [MPSGraph new];
  cache.input = [cache.graph placeholderWithShape:shape2(batch, in_features) dataType:MPSDataTypeBFloat16 name:@"linear_input"];
  cache.weight = [cache.graph placeholderWithShape:shape2(out_features, in_features) dataType:MPSDataTypeBFloat16 name:@"linear_weight"];
  MPSGraphTensor* weight_t = [cache.graph transposeTensor:cache.weight dimension:0 withDimension:1 name:@"linear_weight_t"];
  MPSGraphTensor* out = [cache.graph matrixMultiplicationWithPrimaryTensor:cache.input secondaryTensor:weight_t name:@"linear_matmul"];
  cache.out = [cache.graph castTensor:out toType:MPSDataTypeBFloat16 name:@"linear_out_bf16"];
}

void ensure_legacy_gelu_graph(MPSShape* shape, LegacyGeluGraphCache& cache) {
  if (cache.graph != nil) {
    return;
  }

  // Reproduce PyTorch 2.12's MPSGraph GELU operation order exactly. Newer
  // PyTorch releases use a native Metal kernel whose mathematically equivalent
  // rounding changes diffusion trajectories from the same seed.
  cache.graph = [MPSGraph new];
  cache.input = [cache.graph placeholderWithShape:shape dataType:MPSDataTypeBFloat16 name:@"legacy_gelu_input"];
  constexpr float kSqrtOneHalf = 0.707106781186547524400844362104849039f;
  MPSGraphTensor* sqrt_one_half =
      [cache.graph constantWithScalar:kSqrtOneHalf shape:@[@1] dataType:MPSDataTypeBFloat16];
  MPSGraphTensor* one = [cache.graph constantWithScalar:1.0f shape:@[@1] dataType:MPSDataTypeBFloat16];
  MPSGraphTensor* half = [cache.graph constantWithScalar:0.5f shape:@[@1] dataType:MPSDataTypeBFloat16];
  MPSGraphTensor* cdf = [cache.graph multiplicationWithPrimaryTensor:cache.input
                                                    secondaryTensor:sqrt_one_half
                                                               name:@"legacy_gelu_scaled"];
  cdf = [cache.graph erfWithTensor:cdf name:@"legacy_gelu_erf"];
  cdf = [cache.graph additionWithPrimaryTensor:cdf secondaryTensor:one name:@"legacy_gelu_plus_one"];
  cdf = [cache.graph multiplicationWithPrimaryTensor:cdf secondaryTensor:half name:@"legacy_gelu_half"];
  cache.out = [cache.graph multiplicationWithPrimaryTensor:cdf
                                           secondaryTensor:cache.input
                                                      name:@"legacy_gelu_output"];
}

void ensure_legacy_normal_graph(MPSShape* shape, LegacyNormalGraphCache& cache) {
  if (cache.graph != nil) {
    return;
  }

  // PyTorch 2.12 generated MPS normal samples through MPSGraph. PyTorch 2.13+
  // uses a native Metal Philox/Box-Muller kernel, so equal numeric seeds no
  // longer produce equal tensors. Preserve the former graph and state layout.
  cache.graph = [MPSGraph new];
  cache.state = [cache.graph placeholderWithShape:@[@(at::mps::detail::PHILOX_STATE_N)]
                                           dataType:MPSDataTypeInt32
                                               name:@"legacy_normal_state"];
  MPSGraphRandomOpDescriptor* descriptor =
      [MPSGraphRandomOpDescriptor descriptorWithDistribution:MPSGraphRandomDistributionNormal
                                                    dataType:MPSDataTypeFloat32];
  descriptor.mean = 0.0f;
  descriptor.standardDeviation = 1.0f;
  NSArray<MPSGraphTensor*>* outputs = [cache.graph randomTensorWithShape:shape
                                                              descriptor:descriptor
                                                             stateTensor:cache.state
                                                                    name:@"legacy_normal"];
  cache.out = outputs[0];
}

MPSGraphTensor* linear_2048(MPSGraph* graph, MPSGraphTensor* input, MPSGraphTensor* weight, NSString* name) {
  MPSGraphTensor* weight_t = [graph transposeTensor:weight dimension:0 withDimension:1 name:[name stringByAppendingString:@"_weight_t"]];
  MPSGraphTensor* result = [graph matrixMultiplicationWithPrimaryTensor:input secondaryTensor:weight_t name:name];
  return [graph castTensor:result toType:MPSDataTypeBFloat16 name:[name stringByAppendingString:@"_bf16"]];
}

MPSGraphTensor* rms_norm_heads(MPSGraph* graph, MPSGraphTensor* input, MPSGraphTensor* weight, NSString* name) {
  MPSGraphTensor* input_fp32 = [graph castTensor:input toType:MPSDataTypeFloat32 name:[name stringByAppendingString:@"_fp32"]];
  MPSGraphTensor* squared = [graph squareWithTensor:input_fp32 name:[name stringByAppendingString:@"_squared"]];
  MPSGraphTensor* mean = [graph meanOfTensor:squared axes:@[@(-1)] name:[name stringByAppendingString:@"_mean"]];
  MPSGraphTensor* epsilon = [graph constantWithScalar:1.0e-6 dataType:MPSDataTypeFloat32];
  MPSGraphTensor* variance = [graph additionWithPrimaryTensor:mean secondaryTensor:epsilon name:[name stringByAppendingString:@"_variance"]];
  MPSGraphTensor* inverse_rms = [graph reciprocalSquareRootWithTensor:variance name:[name stringByAppendingString:@"_inverse_rms"]];
  MPSGraphTensor* normalized = [graph multiplicationWithPrimaryTensor:input_fp32 secondaryTensor:inverse_rms name:[name stringByAppendingString:@"_normalized"]];
  MPSGraphTensor* weight_fp32 = [graph castTensor:weight toType:MPSDataTypeFloat32 name:[name stringByAppendingString:@"_weight_fp32"]];
  MPSGraphTensor* scaled = [graph multiplicationWithPrimaryTensor:normalized secondaryTensor:weight_fp32 name:[name stringByAppendingString:@"_scaled"]];
  return [graph castTensor:scaled toType:MPSDataTypeBFloat16 name:[name stringByAppendingString:@"_bf16"]];
}

MPSGraphTensor* apply_rope(MPSGraph* graph, MPSGraphTensor* input, MPSGraphTensor* rope, NSString* name) {
  MPSGraphTensor* input_fp32 = [graph castTensor:input toType:MPSDataTypeFloat32 name:[name stringByAppendingString:@"_input_fp32"]];
  MPSGraphTensor* first = [graph sliceTensor:input_fp32 dimension:3 start:0 length:kHeadDim / 2 name:[name stringByAppendingString:@"_first"]];
  MPSGraphTensor* second = [graph sliceTensor:input_fp32 dimension:3 start:kHeadDim / 2 length:kHeadDim / 2 name:[name stringByAppendingString:@"_second"]];
  MPSGraphTensor* rope_flat = [graph reshapeTensor:rope withShape:@[@1, @(-1), @1, @(kHeadDim / 2), @4] name:[name stringByAppendingString:@"_rope_flat"]];
  MPSGraphTensor* f0 = [graph sliceTensor:rope_flat dimension:4 start:0 length:1 name:[name stringByAppendingString:@"_f0"]];
  MPSGraphTensor* f1 = [graph sliceTensor:rope_flat dimension:4 start:1 length:1 name:[name stringByAppendingString:@"_f1"]];
  MPSGraphTensor* f2 = [graph sliceTensor:rope_flat dimension:4 start:2 length:1 name:[name stringByAppendingString:@"_f2"]];
  MPSGraphTensor* f3 = [graph sliceTensor:rope_flat dimension:4 start:3 length:1 name:[name stringByAppendingString:@"_f3"]];
  f0 = [graph reshapeTensor:f0 withShape:@[@1, @(-1), @1, @(kHeadDim / 2)] name:nil];
  f1 = [graph reshapeTensor:f1 withShape:@[@1, @(-1), @1, @(kHeadDim / 2)] name:nil];
  f2 = [graph reshapeTensor:f2 withShape:@[@1, @(-1), @1, @(kHeadDim / 2)] name:nil];
  f3 = [graph reshapeTensor:f3 withShape:@[@1, @(-1), @1, @(kHeadDim / 2)] name:nil];
  MPSGraphTensor* rotated_first = [graph additionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:f0 secondaryTensor:first name:nil]
                                                      secondaryTensor:[graph multiplicationWithPrimaryTensor:f1 secondaryTensor:second name:nil]
                                                                 name:[name stringByAppendingString:@"_rotated_first"]];
  MPSGraphTensor* rotated_second = [graph additionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:f2 secondaryTensor:first name:nil]
                                                       secondaryTensor:[graph multiplicationWithPrimaryTensor:f3 secondaryTensor:second name:nil]
                                                                  name:[name stringByAppendingString:@"_rotated_second"]];
  MPSGraphTensor* combined = [graph concatTensor:rotated_first withTensor:rotated_second dimension:3 name:[name stringByAppendingString:@"_combined"]];
  return [graph castTensor:combined toType:MPSDataTypeBFloat16 name:[name stringByAppendingString:@"_bf16"]];
}

void ensure_fused_self_attention_graph(int64_t batch, int64_t sequence, FusedSelfAttentionGraphCache& cache) {
  if (cache.graph != nil) {
    return;
  }
  cache.batch = batch;
  cache.sequence = sequence;
  cache.graph = [MPSGraph new];
  cache.input = [cache.graph placeholderWithShape:shape3(batch, sequence, kFlatDim) dataType:MPSDataTypeBFloat16 name:@"input"];
  cache.rope = [cache.graph placeholderWithShape:rope_shape(sequence) dataType:MPSDataTypeFloat32 name:@"rope"];
  cache.q_weight = [cache.graph placeholderWithShape:shape2(kFlatDim, kFlatDim) dataType:MPSDataTypeBFloat16 name:@"q_weight"];
  cache.k_weight = [cache.graph placeholderWithShape:shape2(kFlatDim, kFlatDim) dataType:MPSDataTypeBFloat16 name:@"k_weight"];
  cache.v_weight = [cache.graph placeholderWithShape:shape2(kFlatDim, kFlatDim) dataType:MPSDataTypeBFloat16 name:@"v_weight"];
  cache.output_weight = [cache.graph placeholderWithShape:shape2(kFlatDim, kFlatDim) dataType:MPSDataTypeBFloat16 name:@"output_weight"];
  cache.q_norm_weight = [cache.graph placeholderWithShape:shape1(kHeadDim) dataType:MPSDataTypeBFloat16 name:@"q_norm_weight"];
  cache.k_norm_weight = [cache.graph placeholderWithShape:shape1(kHeadDim) dataType:MPSDataTypeBFloat16 name:@"k_norm_weight"];

  MPSGraphTensor* q_flat = linear_2048(cache.graph, cache.input, cache.q_weight, @"q_projection");
  MPSGraphTensor* k_flat = linear_2048(cache.graph, cache.input, cache.k_weight, @"k_projection");
  MPSGraphTensor* v_flat = linear_2048(cache.graph, cache.input, cache.v_weight, @"v_projection");
  MPSShape* heads_shape = bshd_shape(batch, sequence, kHeads, kHeadDim);
  MPSGraphTensor* q = [cache.graph reshapeTensor:q_flat withShape:heads_shape name:@"q_heads"];
  MPSGraphTensor* k = [cache.graph reshapeTensor:k_flat withShape:heads_shape name:@"k_heads"];
  MPSGraphTensor* v = [cache.graph reshapeTensor:v_flat withShape:heads_shape name:@"v_heads"];
  q = apply_rope(cache.graph, rms_norm_heads(cache.graph, q, cache.q_norm_weight, @"q_norm"), cache.rope, @"q_rope");
  k = apply_rope(cache.graph, rms_norm_heads(cache.graph, k, cache.k_norm_weight, @"k_norm"), cache.rope, @"k_rope");
  MPSGraphTensor* q_bhsd = [cache.graph transposeTensor:q dimension:1 withDimension:2 name:@"q_bhsd"];
  MPSGraphTensor* k_bhsd = [cache.graph transposeTensor:k dimension:1 withDimension:2 name:@"k_bhsd"];
  MPSGraphTensor* v_bhsd = [cache.graph transposeTensor:v dimension:1 withDimension:2 name:@"v_bhsd"];
  MPSGraphTensor* attention = build_attention_output(cache.graph, q_bhsd, k_bhsd, v_bhsd, AttentionVariant::SDPA);
  MPSGraphTensor* attention_bshd = [cache.graph transposeTensor:attention dimension:1 withDimension:2 name:@"attention_bshd"];
  MPSGraphTensor* attention_flat = [cache.graph reshapeTensor:attention_bshd withShape:shape3(batch, sequence, kFlatDim) name:@"attention_flat"];
  cache.out = linear_2048(cache.graph, attention_flat, cache.output_weight, @"output_projection");
}

MPSGraphTensorData* tensor_data_for(const at::Tensor& tensor, MPSShape* shape) {
  id<MTLBuffer> buffer = at::native::mps::getMTLBufferStorage(tensor);
  TORCH_CHECK(buffer != nil, "MPS tensor has no MTLBuffer storage");
  return [[[MPSGraphTensorData alloc] initWithMTLBuffer:buffer shape:shape dataType:MPSDataTypeBFloat16] autorelease];
}

MPSGraphTensorData* float_tensor_data_for(const at::Tensor& tensor, MPSShape* shape) {
  id<MTLBuffer> buffer = at::native::mps::getMTLBufferStorage(tensor);
  TORCH_CHECK(buffer != nil, "MPS tensor has no MTLBuffer storage");
  return [[[MPSGraphTensorData alloc] initWithMTLBuffer:buffer shape:shape dataType:MPSDataTypeFloat32] autorelease];
}

at::Tensor mpsgraph_attention(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const std::string& variant_string) {
  check_attention_tensor(q, "q");
  check_attention_tensor(k, "k");
  check_attention_tensor(v, "v");
  check_attention_shapes(q, k, v);
  AttentionVariant variant = parse_variant(variant_string);

  const int64_t batch = q.size(0);
  const int64_t q_sequence = q.size(1);
  const int64_t kv_sequence = k.size(1);
  const int64_t heads = q.size(2);
  const int64_t head_dim = q.size(3);
  at::Tensor output = at::empty({batch, q_sequence, heads * head_dim}, q.options());
  TORCH_CHECK(output.is_contiguous(), "internal output tensor must be contiguous");

  @autoreleasepool {
    std::lock_guard<std::mutex> guard(g_graph_mutex);
    const std::string key = attention_cache_key(variant, batch, q_sequence, kv_sequence, heads, head_dim);
    AttentionGraphCache& cache = g_attention_caches[key];
    ensure_graph(variant, cache, batch, q_sequence, kv_sequence, heads, head_dim);

    MPSShape* q_shape = bshd_shape(batch, q_sequence, heads, head_dim);
    MPSShape* kv_shape = bshd_shape(batch, kv_sequence, heads, head_dim);
    MPSShape* out_shape = output_graph_shape(batch, q_sequence, heads, head_dim);
    NSDictionary* feeds = @{
      cache.q : tensor_data_for(q, q_shape),
      cache.k : tensor_data_for(k, kv_shape),
      cache.v : tensor_data_for(v, kv_shape),
    };
    NSDictionary* results = @{
      cache.out : tensor_data_for(output, out_shape),
    };
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(stream != nullptr, "Could not get current MPS stream");
    at::native::mps::runMPSGraph(stream, cache.graph, feeds, results);
  }

  return output;
}

at::Tensor mpsgraph_linear(const at::Tensor& input, const at::Tensor& weight) {
  check_linear_tensor(input, weight);

  const int64_t batch = input.size(0);
  const int64_t in_features = input.size(1);
  const int64_t out_features = weight.size(0);
  at::Tensor output = at::empty({batch, out_features}, input.options());
  TORCH_CHECK(output.is_contiguous(), "internal linear output tensor must be contiguous");

  @autoreleasepool {
    std::lock_guard<std::mutex> guard(g_graph_mutex);
    const std::string key = linear_cache_key(batch, in_features, out_features);
    LinearGraphCache& cache = g_linear_caches[key];
    ensure_linear_graph(batch, in_features, out_features, cache);

    NSDictionary* feeds = @{
      cache.input : tensor_data_for(input, shape2(batch, in_features)),
      cache.weight : tensor_data_for(weight, shape2(out_features, in_features)),
    };
    NSDictionary* results = @{
      cache.out : tensor_data_for(output, shape2(batch, out_features)),
    };
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(stream != nullptr, "Could not get current MPS stream");
    at::native::mps::runMPSGraph(stream, cache.graph, feeds, results);
  }

  return output;
}

at::Tensor mpsgraph_legacy_gelu(const at::Tensor& input) {
  TORCH_CHECK(input.is_mps(), "legacy GELU input must be an MPS tensor");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "legacy GELU input must be torch.bfloat16");
  TORCH_CHECK(input.is_contiguous(), "legacy GELU input must be contiguous");
  TORCH_CHECK(input.storage_offset() == 0, "legacy GELU input must have storage_offset=0");

  at::Tensor output = at::empty_like(input);
  @autoreleasepool {
    std::lock_guard<std::mutex> guard(g_graph_mutex);
    const std::string key = shape_string(input);
    LegacyGeluGraphCache& cache = g_legacy_gelu_caches[key];
    MPSShape* shape = tensor_shape(input);
    ensure_legacy_gelu_graph(shape, cache);
    NSDictionary* feeds = @{
      cache.input : tensor_data_for(input, shape),
    };
    NSDictionary* results = @{
      cache.out : tensor_data_for(output, shape),
    };
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(stream != nullptr, "Could not get current MPS stream");
    at::native::mps::runMPSGraph(stream, cache.graph, feeds, results);
  }
  return output;
}

at::Tensor mpsgraph_legacy_normal(
    const std::vector<int64_t>& output_shape,
    std::optional<at::Generator> generator) {
  TORCH_CHECK(!output_shape.empty(), "legacy normal shape must not be empty");
  for (int64_t dimension : output_shape) {
    TORCH_CHECK(dimension >= 0, "legacy normal shape contains a negative dimension");
  }
  auto* mps_generator = at::get_generator_or_default<at::MPSGeneratorImpl>(
      generator,
      at::mps::detail::getDefaultMPSGenerator());
  at::Tensor output = at::empty(output_shape, at::TensorOptions().device(at::kMPS).dtype(at::kFloat));

  @autoreleasepool {
    std::lock_guard<std::mutex> graph_guard(g_graph_mutex);
    const std::string key = shape_string(output);
    LegacyNormalGraphCache& cache = g_legacy_normal_caches[key];
    MPSShape* shape = tensor_shape(output);
    ensure_legacy_normal_graph(shape, cache);

    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(stream != nullptr, "Could not get current MPS stream");
    MPSNDArrayDescriptor* state_descriptor =
        [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeInt32
                                               shape:@[@(at::mps::detail::PHILOX_STATE_N)]];
    MPSNDArray* state_array = [[[MPSNDArray alloc] initWithDevice:stream->device()
                                                       descriptor:state_descriptor] autorelease];
    {
      std::lock_guard<std::mutex> generator_guard(mps_generator->mutex_);
      mps_generator->update_philox_counters();
      [state_array writeBytes:mps_generator->state_data() strideBytes:nil];
    }
    MPSGraphTensorData* state_data = [[[MPSGraphTensorData alloc] initWithMPSNDArray:state_array] autorelease];
    NSDictionary* feeds = @{
      cache.state : state_data,
    };
    NSDictionary* results = @{
      cache.out : float_tensor_data_for(output, shape),
    };
    at::native::mps::runMPSGraph(stream, cache.graph, feeds, results);
  }
  return output;
}

at::Tensor mpsgraph_self_attention(
    const at::Tensor& input,
    const at::Tensor& rope,
    const at::Tensor& q_weight,
    const at::Tensor& k_weight,
    const at::Tensor& v_weight,
    const at::Tensor& output_weight,
    const at::Tensor& q_norm_weight,
    const at::Tensor& k_norm_weight) {
  check_fused_self_attention_tensors(input, rope, q_weight, k_weight, v_weight, output_weight, q_norm_weight, k_norm_weight);
  const int64_t batch = input.size(0);
  const int64_t sequence = input.size(1);
  at::Tensor output = at::empty({batch, sequence, kFlatDim}, input.options());

  @autoreleasepool {
    std::lock_guard<std::mutex> guard(g_graph_mutex);
    const std::string key = std::to_string(batch) + "x" + std::to_string(sequence);
    FusedSelfAttentionGraphCache& cache = g_fused_self_attention_caches[key];
    ensure_fused_self_attention_graph(batch, sequence, cache);
    NSDictionary* feeds = @{
      cache.input : tensor_data_for(input, shape3(batch, sequence, kFlatDim)),
      cache.rope : float_tensor_data_for(rope, rope_shape(sequence)),
      cache.q_weight : tensor_data_for(q_weight, shape2(kFlatDim, kFlatDim)),
      cache.k_weight : tensor_data_for(k_weight, shape2(kFlatDim, kFlatDim)),
      cache.v_weight : tensor_data_for(v_weight, shape2(kFlatDim, kFlatDim)),
      cache.output_weight : tensor_data_for(output_weight, shape2(kFlatDim, kFlatDim)),
      cache.q_norm_weight : tensor_data_for(q_norm_weight, shape1(kHeadDim)),
      cache.k_norm_weight : tensor_data_for(k_norm_weight, shape1(kHeadDim)),
    };
    NSDictionary* results = @{
      cache.out : tensor_data_for(output, shape3(batch, sequence, kFlatDim)),
    };
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(stream != nullptr, "Could not get current MPS stream");
    at::native::mps::runMPSGraph(stream, cache.graph, feeds, results);
  }
  return output;
}

at::Tensor mpsgraph_sdpa(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
  return mpsgraph_attention(q, k, v, "mpsgraph-sdpa");
}

at::Tensor mpsgraph_softmax_api(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
  return mpsgraph_attention(q, k, v, "mpsgraph-softmax-api");
}

at::Tensor mpsgraph_manual_softmax(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
  return mpsgraph_attention(q, k, v, "mpsgraph-manual-softmax");
}

py::dict probe_tensor(const at::Tensor& tensor) {
  py::dict result;
  result["device"] = tensor.device().str();
  result["dtype"] = c10::toString(tensor.scalar_type());
  result["shape"] = py::cast(tensor.sizes().vec());
  result["strides"] = py::cast(tensor.strides().vec());
  result["is_mps"] = tensor.is_mps();
  result["is_contiguous"] = tensor.is_contiguous();
  result["storage_offset"] = tensor.storage_offset();
  result["element_size"] = tensor.element_size();
  result["numel"] = tensor.numel();
  result["storage_data"] = reinterpret_cast<uintptr_t>(tensor.storage().data());

  if (tensor.is_mps() && tensor.has_storage()) {
    id<MTLBuffer> buffer = at::native::mps::getMTLBufferStorage(tensor);
    result["mtlbuffer_available"] = buffer != nil;
    result["mtlbuffer_address"] = reinterpret_cast<uintptr_t>(buffer);
    result["mtlbuffer_length"] = buffer == nil ? 0 : static_cast<uint64_t>([buffer length]);
  } else {
    result["mtlbuffer_available"] = false;
  }
  return result;
}

py::dict probe_runtime() {
  py::dict result;
  @autoreleasepool {
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    result["has_current_mps_stream"] = stream != nullptr;
    if (stream != nullptr) {
      id<MTLCommandQueue> queue = stream->commandQueue();
      result["has_command_queue"] = queue != nil;
      result["command_queue_address"] = reinterpret_cast<uintptr_t>(queue);
      id<MTLDevice> device = stream->device();
      result["has_device"] = device != nil;
      result["device_name"] = device == nil ? "" : std::string([[device name] UTF8String]);
    }
    result["attention_graph_cache_count"] = py::int_(g_attention_caches.size());
    result["linear_graph_cache_count"] = py::int_(g_linear_caches.size());
    result["legacy_gelu_graph_cache_count"] = py::int_(g_legacy_gelu_caches.size());
    result["legacy_normal_graph_cache_count"] = py::int_(g_legacy_normal_caches.size());
    result["fused_self_attention_graph_cache_count"] = py::int_(g_fused_self_attention_caches.size());
  }
  return result;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Audited PyTorch MPS tensor to MPSGraph attention interop probe";
  m.def("probe_tensor", &probe_tensor, "Inspect MPS tensor storage/MTLBuffer metadata");
  m.def("probe_runtime", &probe_runtime, "Inspect current PyTorch MPS stream/runtime metadata");
  m.def("mpsgraph_attention", &mpsgraph_attention, "Run Anima BF16 self-attention through a shape-keyed MPSGraph variant");
  m.def("mpsgraph_linear", &mpsgraph_linear, "Run a BF16 2D linear matmul through MPSGraph");
  m.def("mpsgraph_legacy_gelu", &mpsgraph_legacy_gelu, "Run byte-compatible PyTorch 2.12 exact GELU through MPSGraph");
  m.def(
      "mpsgraph_legacy_normal",
      &mpsgraph_legacy_normal,
      py::arg("shape"),
      py::arg("generator") = std::nullopt,
      "Generate PyTorch 2.12-compatible MPSGraph standard-normal noise");
  m.def("mpsgraph_self_attention", &mpsgraph_self_attention, "Run fused Anima self-attention through MPSGraph");
  m.def("mpsgraph_sdpa", &mpsgraph_sdpa, "Run Anima BF16 self-attention through MPSGraph");
  m.def("mpsgraph_softmax_api", &mpsgraph_softmax_api, "Run Anima BF16 self-attention through MPSGraph matmul + softmax");
  m.def("mpsgraph_manual_softmax", &mpsgraph_manual_softmax, "Run Anima BF16 self-attention through MPSGraph manual softmax");
}
