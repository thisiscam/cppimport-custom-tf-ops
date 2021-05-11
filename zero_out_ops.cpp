// cppimport

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });


#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output_flat(0) = input(0);
  }
};

REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);

/*
<%
import sys
import tensorflow as tf;
cfg['extra_compile_args'] = ["-fPIC", "-O2", "-std=c++11"] + tf.sysconfig.get_compile_flags()
cfg['extra_link_args'] = ["-shared"] + tf.sysconfig.get_link_flags()
if sys.platform == "darwin":
  cfg['extra_link_args'] += ["-undefined", "dynamic_lookup"]
  # Don't compile bundle on macOS
  from distutils import sysconfig
  vars = sysconfig.get_config_vars()
  vars['LDSHARED'] = vars['LDSHARED'].replace('-bundle', '-dynamiclib')
%>
*/
