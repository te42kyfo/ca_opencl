#ifndef OCL_HPP
#define OCL_HPP

#include <CL/opencl.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "cl_helper.h"

std::string platformInfo(cl_platform_id id, cl_platform_info param_name) {
  size_t size;
  checkOclErrors(clGetPlatformInfo(id, param_name, 0, NULL, &size));
  std::vector<char> infoVec(size);
  checkOclErrors(clGetPlatformInfo(id, param_name, size, infoVec.data(), NULL));
  return std::string(begin(infoVec), end(infoVec));
}

std::string deviceInfo(cl_device_id id, cl_device_info param_name) {
  size_t size;
  checkOclErrors(clGetDeviceInfo(id, param_name, 0, NULL, &size));
  std::vector<char> infoVec(size);
  checkOclErrors(clGetDeviceInfo(id, param_name, size, infoVec.data(), NULL));
  return std::string(begin(infoVec), end(infoVec));
}

class OCL {
 public:
  OCL(unsigned int selection) {
    std::vector<std::pair<cl_platform_id, cl_device_id>> options;
    cl_uint num_platforms;

    checkOclErrors(clGetPlatformIDs(0, NULL, &num_platforms));
    std::vector<cl_platform_id> platforms(num_platforms);
    checkOclErrors(clGetPlatformIDs(num_platforms, platforms.data(), NULL));

    for (auto p : platforms) {
      cl_uint num_devices = 0;
      checkOclErrors(
          clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices));
      if (num_devices != 0) {
        std::vector<cl_device_id> devices(num_devices);
        checkOclErrors(clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, num_devices,
                                      devices.data(), NULL));
        for (auto d : devices) {
          options.push_back(std::make_pair(p, d));
        }
      }
    }

    if (selection >= options.size()) selection = 0;
    for (unsigned int i = 0; i < options.size(); i++) {
      std::cerr << "# ";
      if (i == selection)
        std::cerr << " * ";
      else
        std::cerr << "   ";
      std::cerr << platformInfo(options[i].first, CL_PLATFORM_NAME) << ": "
                << deviceInfo(options[i].second, CL_DEVICE_NAME) << "\n";
    }
    device = options[selection].second;

    cl_int error;
    context = clCreateContext(0, 1, &device, NULL, NULL, &error);
    queue = clCreateCommandQueue(context, device, 0, &error);
    checkOclErrors(error);
  }

  ~OCL() {
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
  }

  cl_kernel buildKernel(std::string filename, std::string kernel_name,
                        std::string additional_options = "") {
    std::string source;
    try {
      std::ifstream t(filename.c_str());
      if (!t) {
        std::cerr << "buildKernel: Could not open " << filename << "\n";
        exit(1);
      }
      source = std::string(std::istreambuf_iterator<char>(t),
                           std::istreambuf_iterator<char>());
    } catch (std::exception& e) {
      exit(1);
    }

    cl_int error = CL_SUCCESS;
    const char* char_source = source.c_str();
    cl_program program =
        clCreateProgramWithSource(context, 1, &char_source, NULL, &error);
    checkOclErrors(error);
    std::string options("");
    options.append(additional_options);
    clBuildProgram(program, 0, NULL, options.c_str(), NULL, NULL);
    cl_build_status build_status = CL_BUILD_SUCCESS;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS,
                          sizeof(cl_build_status), &build_status, nullptr);

    if (build_status != CL_BUILD_SUCCESS) {
      size_t logsize = 0;
      checkOclErrors(clGetProgramBuildInfo(
          program, device, CL_PROGRAM_BUILD_LOG, 0, 0, &logsize));
      char log[logsize];
      checkOclErrors(clGetProgramBuildInfo(
          program, device, CL_PROGRAM_BUILD_LOG, logsize, log, nullptr));
      std::cerr << "Build log of file " << filename << " :\n" << log << "\n";
    }

    cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &error);
    checkOclErrors(error);

    return kernel;
  }

  template <typename T>
  cl_mem createAndUpload(std::vector<T> const& hbuf) {
    cl_int error;
    cl_mem dbuf = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 hbuf.size() * sizeof(T), NULL, &error);
    checkOclErrors(clEnqueueWriteBuffer(queue, dbuf, true, 0,
                                        hbuf.size() * sizeof(T), hbuf.data(), 0,
                                        NULL, NULL));
    return dbuf;
  }

  template <typename T>
  std::vector<T> download(cl_mem dbuf) {
    size_t buf_size;
    checkOclErrors(
        clGetMemObjectInfo(dbuf, CL_MEM_SIZE, sizeof(size_t), &buf_size, NULL));
    std::vector<T> hbuf(buf_size / sizeof(T));

    checkOclErrors(clEnqueueReadBuffer(queue, dbuf, true, 0,
                                       hbuf.size() * sizeof(T), hbuf.data(), 0,
                                       NULL, NULL));
    return hbuf;
  }

  void finish() { clFinish(queue); }

  template <typename... Args>
  void execute(cl_kernel kernel, size_t dim, std::vector<size_t> global_size,
               std::vector<size_t> local_size, Args... args) {
    execute_t(0, kernel, dim, global_size, local_size, args...);
  }

 private:
  void execute_t(size_t argument_index, cl_kernel kernel, size_t dim,
                 std::vector<size_t> global_size,
                 std::vector<size_t> local_size) {
    checkOclErrors(clEnqueueNDRangeKernel(queue, kernel, dim, NULL,
                                          global_size.data(), local_size.data(),
                                          0, NULL, NULL));
  }

  template <typename T, typename... Args>
  void execute_t(size_t argument_index, cl_kernel kernel, size_t dim,
                 std::vector<size_t> global_size,
                 std::vector<size_t> local_size, T argument, Args... args) {
    cl_int err = clSetKernelArg(kernel, argument_index, sizeof(T), &argument);
    if (err != CL_SUCCESS) {
      std::cerr << "Argument index " << argument_index << "\n";
      checkOclErrors(err);
    } else {
      execute_t(argument_index + 1, kernel, dim, global_size, local_size,
                args...);
    }
  }

 public:
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
};

#endif
