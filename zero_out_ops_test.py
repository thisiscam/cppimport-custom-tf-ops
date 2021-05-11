def load_ops(ops_name: str):
  import cppimport
  from tensorflow.python.framework import load_library
  from tensorflow.python.platform import resource_loader

  shared_lib_path = cppimport.build(ops_name)
  return load_library.load_op_library(resource_loader.get_path_to_datafile(shared_lib_path))


zero_out_ops = load_ops("zero_out_ops")
print(zero_out_ops.zero_out(list(range(10, 1, -1))))
