FROM rocm/dev-ubuntu-24.04:6.3

WORKDIR /workspace

RUN apt-get update
RUN apt-get install python3.12 -y
RUN apt-get install python-is-python3 -y
RUN apt-get install pip -y
RUN apt-get install git -y
RUN apt-get install curl -y
RUN apt-get install gdb -y
RUN apt-get install cmake -y

# https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.0/how_to_guides/debugging.html

ENV AMD_LOG_LEVEL=3

# enable specific devices
# ENV HIP_VISIBLE_DEVICES="0,2"

# https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.0/how_to_guides/debugging.html#dump-code-object
# ENV GPU_DUMP_CODE_OBJECT=1