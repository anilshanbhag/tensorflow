// A test NVL op.

#include <iostream>
#include <dlfcn.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

// Get the symbol for NVL's run()
typedef char* (*nvl_main_t)(char* args);

using namespace tensorflow;

REGISTER_OP("Nvl")
    .Attr("nvl_code: string = ''")
    .Attr("output_dims: int32")
    .Attr("Tin: list({int32, float32, float64})")
    .Attr("Tout: list({int32, float32, float64})")
    .Input("in: Tin")
    .Output("out: Tout");

class NvlOp : public OpKernel {
 public:
  explicit NvlOp(OpKernelConstruction* context) : OpKernel(context) {
    // Extract the NVL program
    string nvl_code;
    OP_REQUIRES_OK(context, context->GetAttr("nvl_code", &nvl_code));
    nvl_program_ = str_util::CEscape(nvl_code);
    OP_REQUIRES_OK(context, context->GetAttr("output_dims", &output_dims_));

    // Generate NVL code
    NvlGenerate(nvl_program_, "tftest.nvl");
  }

  void Compute(OpKernelContext* context) override {
    context_ = context;

    // Get input tensors
    OpInputList inputs;
    OP_REQUIRES_OK(context, context->input_list("in", &inputs));

    // Make sure we have manageable inputs.
    for (int i=0; i<inputs.size(); i++) {
      const Tensor& tensor = inputs[i];
      const DataType& dtype = tensor.dtype();
      OP_REQUIRES(context_, dtype == DT_INT32 || dtype == DT_FLOAT ||
          dtype == DT_DOUBLE || dtype == DT_INT64,
          errors::InvalidArgument("Unsupported dtype"));
    }

    // Execute NVL code
    char* result = NvlExecute(inputs);

    // Create the output tensor
    OpOutputList output_list;
    OP_REQUIRES_OK(context, context->output_list("out", &output_list));

    // Convert NVL result back to tensor
    CHECK(NvlVecsToTensors(result, context, &output_list));
  }

 private:
  void NvlGenerate(const string& program, const string& filename) {

    FILE* nvl_file = fopen(filename.c_str(), "w");
    CHECK_NOTNULL(nvl_file);

    CHECK_EQ(fwrite(program.c_str(), program.size(), 1, nvl_file), 1);

    CHECK_EQ(fclose(nvl_file), 0);

    // Build
    CHECK_EQ(system("make dylib"), 0);

    // Load previously generated .so file
    void* handle = dlopen("run.so", RTLD_LAZY);
    CHECK_NOTNULL(handle);

    dlerror();
    nvl_main_ = (nvl_main_t) dlsym(handle, "run");
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
      dlclose(handle);
      LOG(ERROR) << "Failed to load NVL run() symbol!";
    }

/*    // Done, so unload*/
    /*CHECK_EQ(dlclose(handle), 0);*/
  }

  int CalculateBufferSize(const OpInputList& inp) {
    int buf_size = 0;
    for (int i=0; i<inp.size(); i++) {
      const Tensor& tensor = inp[i];
      const DataType& dtype = tensor.dtype();
      const TensorShape& shape = tensor.shape();

      if (shape.dims() > 0) {
        buf_size += 24;
      } else {
        buf_size += GetDTypeSize(dtype);
      }
    }

    return buf_size;
  }

  int GetDTypeSize(const DataType& dtype) {
    int elem_size = 8;
    if (dtype == DT_INT32 || dtype == DT_FLOAT)
      elem_size = 4;
    return elem_size;
  }

  char* NvlExecute(const OpInputList& inputs) {
    // We need to construct the args.
    // Copy int, float directly into buffer.
    // For rest, copy as vector struct
    // [data_pointer:long, size: long, region:long=0].
    int bufferSize = CalculateBufferSize(inputs);
    char* buffer = new char[bufferSize];
    int curPointer = 0;
    for (int i=0; i<inputs.size(); i++) {
      const Tensor& tensor = inputs[i];
      int dtypeSize = GetDTypeSize(tensor.dtype());
      if (tensor.shape().dims() > 0) {
        // Get data. Return is a StringPiece.
        auto data = tensor.tensor_data();
        long region = 0;
        long size = data.size() / dtypeSize;
        char* data_ptr = (char*)data.data();
        memcpy(buffer + curPointer, &data_ptr, 8); curPointer += 8;
        memcpy(buffer + curPointer, &size, 8); curPointer += 8;
        memcpy(buffer + curPointer, &region, 8); curPointer += 8;
      } else {
        auto data = tensor.tensor_data();
        memcpy(buffer + curPointer, data.data(), dtypeSize);
        curPointer += dtypeSize;
      }
    }

    // Invoke NVL program
    char* nvl_result = nvl_main_(buffer);
    CHECK_NOTNULL(nvl_result);
    return nvl_result;
  }

  bool NvlVecsToTensors(char* nvl_data, OpKernelContext* context,
                        OpOutputList* output_list) {
    Tensor* out_tensor;
    TensorShape shape;

    // TODO: Remove data copying.
    if (output_dims_ == 0) {
      if (!output_list->allocate(0, shape, &out_tensor).ok())
        return false;

      auto output = out_tensor->template flat<int>();
      output(0) = *(int*)nvl_data;
    } else if (output_dims_ > 0) {
      long vecSize = *(long*)(nvl_data+8);
      shape.AddDim(vecSize);
      if (!output_list->allocate(0, shape, &out_tensor).ok())
        return false;

      auto output = out_tensor->template flat<int>();
      int* vecData = *(int**)nvl_data;
      for (int i=0; i<vecSize; i++)
        output(i) = vecData[i];
    }

    return true;
  }

  string nvl_program_;

  int output_dims_;

  nvl_main_t nvl_main_;

  OpKernelContext* context_;
};

REGISTER_KERNEL_BUILDER(Name("Nvl").Device(DEVICE_CPU),
                                   NvlOp);
